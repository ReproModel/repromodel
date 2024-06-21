export function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1)
}

export function capitalizeFirstLetterOfEachWord(string) {
    if (string === string.toUpperCase()) {
        return string
    }
    else {
        return string.replace(/(^|[\s-])\S/g, function(match) {
            return match.toUpperCase();
        })
    }
}