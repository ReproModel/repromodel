export function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1)
}

export function capitalizeFirstLetterOfEachWord(string) {
    if (string === string.toUpperCase()) {
        return string
    }
    else {
        return string.replace(/(^|[\s-])\S/g, function(match) {
            return match.toUpperCase()
        })
    }
}

export function capitalizeAndRemoveUnderscore(string) {
    let returnString = string.charAt(0).toUpperCase() + string.slice(1)

    // Handle LR Schedulers
    returnString = returnString.replace("Lr", "LR")

    // Unicode Underscore Character I
    returnString = returnString.replace("_", " ")

    // Unicode Underscore Character II (it's different)
    returnString = returnString.replace("_", " ")

    return capitalizeFirstLetterOfEachWord(returnString)
}