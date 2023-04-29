setTimeout(function mainFunction(){
try {
	document.getElementById(“run”).addEventListener(“click”, function(){
	createPDF();
    }
)}
catch(err) {
	console.log(err)
}
	console.log(‘Listener Added!’);
}, 30000);