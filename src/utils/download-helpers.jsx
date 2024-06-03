export const handleDownload = (content, outputName, type) => {
    let json, blob

    if(type === "json"){
      json = JSON.stringify(content, null, 2);
      blob = new Blob([json], { type: 'application/json' });

    }else{
      console.log("Filetype currently not supported")
      return
    }
   
    const href = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = href;
    link.download = `${outputName}.${type}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
