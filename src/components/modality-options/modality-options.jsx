import "./modality-options.css"

const imageModels = [
    { label: "ResNet", image: "https://production-media.paperswithcode.com/thumbnails/method/6f12f862-c1a9-4278-8ab9-82c658036935.jpg", numPapers: "2157", href: "" },
    { label: "Visual Transformer", image: "https://production-media.paperswithcode.com/thumbnails/method/1ec5160c-aabe-44b6-a463-3db231949c27.jpg", numPapers: "1430", href: "" },
    { label: "VGG", image: "https://production-media.paperswithcode.com/thumbnails/method/method-0000000312-bb51f64c.jpg", numPapers: "488", href: ""},
    { label: "DenseNet", image: "https://production-media.paperswithcode.com/thumbnails/method/method-0000000434-b414a7f9_Ctryexj.jpg", numPapers: "401", href: ""},
    { label: "MobileNetV2", image: "https://production-media.paperswithcode.com/thumbnails/method/method-0000000170-4df6c581_jjvLsL7.jpg", numPapers: "328", href: "" }
  ]

const ModalityOptions = () => {
  
  return (

    <div className = "container">

        <div className = "card-deck card-break">

            { imageModels?.map((model, idx) => (

                <div className = "card" key = { idx }>
                    
                    <a href = { model.href }>

                        <div className = "row">
                            
                            <div className = "col-xl-4">
                                            
                                <div className = "card-background-image" style = { { backgroundImage: "url('" + model.image + "')"  } }/>
        
                            </div>
                            
                            <div className = "col-xl-8 card-title">

                                <p>{ model.label }</p>

                            </div>

                        </div>

                        <div className = "card-body">
                            
                            <div className = "card-subtitle"> { model.numPapers + " papers" }</div>
                            
                        </div>
                    
                    </a>

                </div>

            ))}

        </div>
    
    </div>
  )
}

export default ModalityOptions