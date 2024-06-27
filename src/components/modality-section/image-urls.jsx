const defaultUrl = "url('https://production-media.paperswithcode.com/icons/task/task-0000000228-40138330.jpg')";

const imageUrls = {

    // Task
    "object detection": "url('https://production-media.paperswithcode.com/icons/task/dd004e56-bc49-4cc1-b0d5-186f2dd17ce8.jpg')",
    "instance segmentation": "url('https://production-media.paperswithcode.com/icons/task/task-0000000003-fae0daac_XS6W0G2.jpg')",
    "keypoint detection": "url('https://production-media.paperswithcode.com/icons/task/27efc689-216a-4b18-b27f-dee62097414a.jpg')",
    "classification": "url('https://production-media.paperswithcode.com/icons/task/0aa45ecb-2bb1-4c8d-bd0c-16b4d9de739d.jpg')",
    "segmentation": "url('https://production-media.paperswithcode.com/icons/task/48d55b59-3af2-4a6d-a195-572f1d4a1867.jpg')",
    "regression": "url('https://dezyre.gumlet.io/images/blog/linear-regression-in-machine-learning/image_12967788041695996160174.png')",

    // Subtask
    "binary": "url('https://cdn-images-1.medium.com/max/638/1*5zuXpqSzciVTB_4jKd-54Q.png')",
    "multi-class": "url('https://pub.mdpi-res.com/information/information-14-00333/article_deploy/html/images/information-14-00333-g005.png?1686638895')",
    "image": "url('https://production-media.paperswithcode.com/datasets/0daad4f0-886b-44ed-9b96-80d99e037f16.png')",
    "attribute": defaultUrl,
    "texture": "url('https://www.mdpi.com/applsci/applsci-09-03900/article_deploy/html/images/applsci-09-03900-g003-550.jpg')",
    "semantic": "url('https://production-media.paperswithcode.com/thumbnails/task/0d834282-fd21-4e57-be69-d5c2ed538690.jpg')",
    "instance": "url('https://miro.medium.com/v2/resize:fit:1400/0*URbHfjxhCvjGHE07')",
    "error": "url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS-Aw7MjJTPnJQ_kBk8Sawh2CX-2cVC6wvTOuAKmta60E83qE3hghmWS1ew9iUhq8-JJ0I&usqp=CAU')",

    // Modality
    "images": "url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ9IpuHCgEjrmrr_cEMlm54bJgiIA1Sm80jlw&s')",
    "video": "url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRSzsjFc3ep--2--rQAk0Rm77KFEUxB-_lRjBIZIWwuexp3GBCBXteBpjEPBU2UvtX_-Xw&usqp=CAU')",
    "medical": "url('https://production-media.paperswithcode.com/thumbnails/task/task-0000000876-6fbe75a2_gBlYteG.jpg')",
    "tabular": "url('https://framerusercontent.com/images/L0NPKK0APCQqOI72SAsORmuo08.jpg')",

    // Submodality
    "RGB": "url('https://pbs.twimg.com/profile_images/565252991561641984/KlUtwnF4_400x400.png')",
    "grayscale": "url('https://onlinepngtools.com/images/examples-onlinepngtools/black-white-grid.png')",
    "financial": "url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_Ary2zQQfvR89Kaq2djBX2tPUQDdAmL13XjOoaI3WmtyIsTQlPEnUz6YxTTJw8ANVFxg&usqp=CAU')",
    "MRI": "url('https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs00247-022-05478-5/MediaObjects/247_2022_5478_Fig10_HTML.png')",
    "CT": "url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSor-Uyr3NmRTy-Leb_3-X1bCtBHHzdHIMwxQ&s')"
}

export { imageUrls }