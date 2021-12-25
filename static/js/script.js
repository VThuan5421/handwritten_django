
var loadFile = function(event) {
	const image = document.getElementById('output');
	image.src = URL.createObjectURL(event.target.files[0]);
};

/* wait for the content of the window element to load, the performs
the operations. This is considered best practice */
window.addEventListener('load', () => {
	resize(); // resizes the canvas
	document.addEventListener('mousedown', startPainting);
	document.addEventListener('mouseup', stopPainting);
	document.addEventListener('mousemove', sketch);
	window.addEventListener('resize', resize)
});

const canvas = document.getElementById("canvas")
canvas.style.background = "black";
// context for the canvas for 2 dimesional operations
const ctx = canvas.getContext('2d')
// Resize the canvas to the availabel size of the window
function resize() {
	ctx.canvas.width = 300;
	ctx.canvas.height = 300;
}
// stores the initial position of the cursor
let coord = {x: 0, y: 0};
// This is the flag that we are going to use to trigger drawing
let paint = false;

// Updates the coordinates of the cursor when and event e is triggered
// to the coordinates where the said event is triggered
function getPosition(event) {
	coord.x = event.clientX - canvas.offsetLeft;
	coord.y = event.clientY - canvas.offsetTop;
}

// The following functions toggle the flag to start and stop drawing
function startPainting(event) {
	paint = true;
	getPosition(event);
}
function stopPainting() {
	paint = false;
}
function sketch(event) {
	if (!paint) return;
	ctx.beginPath();

	ctx.lineWidth = 12;
	
	// Sets the end of the lines drawn to a round shape;
	ctx.linecap = 'round'

	ctx.strokeStyle = 'white';

	// The cursor to start drawing moves to this coordinate
	ctx.moveTo(coord.x, coord.y);
	// The position of the cursor gets updated as we move the mouse around.
	getPosition(event);

	// A line is traced from start coordinate to this coordinate
	ctx.lineTo(coord.x, coord.y);
	// Draws the line.
	ctx.stroke();
}
function clearCanvas() {
	//document.getElementById('clear').addEventListener('click', function() {
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	//}, false);
}

var image = new Image();
image.id = "pic"
image.src = canvas.toDataURL();
document.getElementById('image_for_crop').appendChild(image);