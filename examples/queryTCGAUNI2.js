import sharp from "sharp";
import FormData from "form-data";
import axios from "axios";
import fs from "fs";

// NodeJS example of querying a tiff patch (kidney)

async function queryHist2Query(
    tiffPath,
    x,
    y,
    width,
    height
) {

    const pngBuffer = await sharp(tiffPath)
        .extract({
            left: x,
            top: y,
            width: width,
            height: height
        })
        .png()
        .toBuffer();


    const form = new FormData();

    form.append("k", 5);
    form.append("url", "true");

     form.append(
        "patch",
        pngBuffer,
        {
            filename: "patch.png",
            contentType: "image/png"
        }
    );


    const response = await axios.post(
        "http://localhost:7000/search",
        form,
        {
            headers: form.getHeaders(),
            timeout: 300000
        }
    );


    return response.data;
}


const result = await queryHist2Query(
    "./kidney_crop.tif",
    0,
    0,
    224 * 4,
    224 * 4);

console.log(result);