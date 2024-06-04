import React from "react"
import ModalityOptions from "./modality-options"
import { Typography } from "@mui/material"
import "./modality-options.css"

const imageModels = [
    { label: "ResNet", image: "https://production-media.paperswithcode.com/thumbnails/method/6f12f862-c1a9-4278-8ab9-82c658036935.jpg", numPapers: "2157", href: "" },
    { label: "Visual Transformer", image: "https://production-media.paperswithcode.com/thumbnails/method/1ec5160c-aabe-44b6-a463-3db231949c27.jpg", numPapers: "1430", href: "" },
    { label: "VGG", image: "https://production-media.paperswithcode.com/thumbnails/method/method-0000000312-bb51f64c.jpg", numPapers: "488", href: ""},
    { label: "DenseNet", image: "https://production-media.paperswithcode.com/thumbnails/method/method-0000000434-b414a7f9_Ctryexj.jpg", numPapers: "401", href: ""},
    { label: "MobileNetV2", image: "https://production-media.paperswithcode.com/thumbnails/method/method-0000000170-4df6c581_jjvLsL7.jpg", numPapers: "328", href: "" }
  ]

  const Modalities = [
    { label: "Image", image: "https://production-media.paperswithcode.com/thumbnails/task/task-0000000509-66402dc1_C47uozM.jpg", numPapers: "2157 Datasets", href: "" },
    { label: "Video", image: "https://production-media.paperswithcode.com/thumbnails/task/6f9c5c9e-b5fc-4ce3-b423-a3b196f0252c.jpg", numPapers: "2157 Datasets", href: "" },
    { label: "Audio", image: "https://production-media.paperswithcode.com/thumbnails/method/method-0000000312-bb51f64c.jpg", numPapers: "2157 Datasets", href: ""},
    { label: "3D", image: "https://production-media.paperswithcode.com/icons/task/48d55b59-3af2-4a6d-a195-572f1d4a1867.jpg", numPapers: "2157 Datasets", href: ""},
    { label: "Text", image: "https://production-media.paperswithcode.com/icons/task/f3b5b381-ae14-4572-a44b-79c4aea62230.jpg", numPapers: "2157 Datasets", href: ""},
    
  ]

  const Tasks = [
    { label: "Semantic Segmentation", image: "https://production-media.paperswithcode.com/icons/task/b45b7a24-e2dd-47e2-9d1f-0f372e5d9074.jpg", numPapers: "180 models", href: "" },
    { label: "Classification", image: "https://production-media.paperswithcode.com/icons/task/0aa45ecb-2bb1-4c8d-bd0c-16b4d9de739d.jpg", numPapers: "180 models", href: "" },
    { label: "Object Detecttion", image: "https://production-media.paperswithcode.com/icons/task/dd004e56-bc49-4cc1-b0d5-186f2dd17ce8.jpg", numPapers: "180 models", href: ""},
 
  ]

const ModalitySection = () => {
  
  return (

    <div className = "container">
        <Typography style={{marginBottom: "16px", marginTop: "16px"}} variant="h4"> Choose your Modality </Typography>
        <ModalityOptions imageModels={Modalities}/>
        <Typography style={{marginBottom: "16px", marginTop: "64px"}} variant="h4"> What Task to perform?  </Typography>
        <ModalityOptions imageModels={Tasks}/>

    
    </div>
  )
}

export default ModalitySection