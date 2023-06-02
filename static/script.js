function recordAudio() {
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = function(event) {
        var result = event.results[0][0].transcript;
        document.getElementById('input_text').value = result;
    };

    recognition.start();
}

