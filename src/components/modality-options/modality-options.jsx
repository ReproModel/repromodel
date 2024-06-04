import "./modality-options.css"



const ModalityOptions = ({imageModels}) => {
  
  return (

    

        <div className = "card-deck card-break">

            { imageModels?.map((model, idx) => (

                <div className = "card" key = { idx }>
                    
                    <a href = { model.href }>

                        <div className = "row">
                            
                            <div className = "card-image-container">
                                            
                                <div className = "card-image" style = { { backgroundImage: "url('" + model.image + "')"  } }/>
        
                            </div>
                            
                            <div className = "card-title">

                                <p>{ model.label }</p>

                            </div>

                        </div>

                        <div className = "card-body">
                            
                            <div className = "card-subtitle"> { model.numPapers }</div>
                            
                        </div>
                    
                    </a>

                </div>

            ))}

        </div>
    
    
  )
}

export default ModalityOptions