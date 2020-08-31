const express = require('express');
const { exec, execSync } = require("child_process");
const multer = require('multer');
const path = require('path');
const app = express()
const port = 14444
require('log-timestamp');

// read static html
const fs = require('fs');
var index_page_html_string = fs.readFileSync('./html/index.html');


// listener for decode output
const chokidar = require('chokidar');

// http response handling
app.use(express.static('html'));

app.get('/', (req, res) => {
    res.statusCode = 200;
    res.send(index_page_html_string)
})

const upload = multer({ dest: 'uploads/' });
const rawmp3dir = path.join(__dirname, "../model_export/feat_extract/preprocdir/rawmp3");
const featscppath = path.join(__dirname, "../model_export/feat_extract/preprocdir/dump/feats.scp")
// handle uploads
app.post('/upload-wav', upload.single('uploadfile'), (req, res) => {
    console.log(`new upload: ${req.file.filename}`);
    console.log(`filename: ${req.body.uploadfilename}`);
    // delete all files in rawmp3s directory
    fs.readdir(rawmp3dir, (err, files) => {
        if (err) throw err;
        for (const file of files) {
            fs.unlink(path.join(rawmp3dir, file), err => {
                if (err) throw err;
                console.log("old files deleted")
                // generate random prefix
                var prefix = ""
                var chars = "0123456789"
                for (var i = 0; i < 10; i++) {
                    prefix += chars.charAt(Math.floor(Math.random() * 10));
                }
                full_new_filename = prefix + req.body.uploadfilename
                fs.rename(path.join("uploads", req.file.filename), path.join(rawmp3dir, full_new_filename), (err) => {
                    if (err) throw err;
                    console.log("uploaded file moved to rawmp3")
                    monitor_file_name = full_new_filename + "_decode_output.txt";
                    fs.writeFileSync(monitor_file_name, "");
                    console.log(`Watching for file changes on ${monitor_file_name}`);
                    // listen for decoding output
                    const watcher = chokidar.watch(monitor_file_name)
                    watcher.on('all', (event, path) => {
                        if ("change" == event) {
                            fs.readFile(monitor_file_name, "utf8", (err, contents) => {
                                console.log(contents)
                                res.send(contents);
                                fs.unlink(monitor_file_name, err => {
                                    if (err) throw err;
                                })
                            })
                            watcher.close().then(() => console.log(`watcher for ${monitor_file_name} closed`));
                        }
                    });
                    // start decoding
                    exec("./decode.sh", (error, stdout, stderr) => {
                        if (error) {
                            console.log(`error: ${error.message}`);
                            return;
                        }
                        if (stderr) {
                            console.log(`stderr: ${stderr}`);
                            return;
                        }
                        console.log(stdout);
                    });
                })
            })
        }
    })
});


app.listen(port, () => {
    console.log(`Listening at http://localhost:${port}`)
})