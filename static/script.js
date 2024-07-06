async function fnc() {
    const tweet = document.getElementById('text').value;
    const resultDiv = document.getElementById('resultText');

    const emojis = ["😞","😄","😍","😡","😱","😲"]

    // Afficher un message d'attente
    resultDiv.innerHTML = 'Analyse en cours...';

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: tweet }),
        });

        if (!response.ok) {
            throw new Error('Erreur de réseau');
        }

        const data = await response.json();
        const index = data.index;
        resultDiv.innerHTML = `<p >Sentiment : ${data.result} <span style="font-size:1.8rem;">${emojis[index]}</span></p> `;
    } catch (error) {
        resultDiv.innerHTML =' <p style="color: red;">Erreur : ${error.message}</p>';
    }
}

async function Trendfnc() {
    const tweet = document.getElementById('text').value;



    // Afficher un message d'attente
    resultDiv.innerHTML = 'Analyse en cours...';

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: tweet }),
        });

        if (!response.ok) {
            throw new Error('Erreur de réseau');
        }

        const data = await response.json();
        const index = data.index;
        resultDiv.innerHTML = `<p >Sentiment : ${data.result} <span style="font-size:1.8rem;">${emojis[index]}</span></p> `;
    } catch (error) {
        resultDiv.innerHTML =' <p style="color: red;">Erreur : ${error.message}</p>';
    }
}