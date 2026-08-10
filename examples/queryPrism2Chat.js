import sharp from "sharp";
import FormData from "form-data";
import axios from "axios";
import fs from "fs";

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

    form.append(
        "patch",
        pngBuffer,
        {
            filename: "patch.png",
            contentType: "image/png"
        }
    );

    form.append("question", "What is this tissue?");
    form.append("max_token_response", 100);


    const response = await axios.post(
        "http://localhost:7000/chat",
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