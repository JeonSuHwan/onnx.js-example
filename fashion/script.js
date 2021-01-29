const CANVAS_SIZE = 280;
const CANVAS_SCALE = 1.0;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const option = document.getElementById("imageSelect");

// Load our model.
const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./ffmodel.onnx");

var fashion = ["T-shirt", "Trouser", "Pullover", "Dress",
  "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Angkle boot"
];

ctx.fillStyle = "#212121";
ctx.font = '28px sans-serif';
ctx.textAlign = "center"
ctx.textBaseLine = "middle";
ctx.fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

// draw images
function imageKindChange() {
  img_source = option.value;
  var img = new Image();
  //ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  img.addEventListener('load', function() {
    ctx.drawImage(img, 0, 0);
    updatePrediction();
  }, false);
  img.src = img_source
}

async function updatePrediction() {
  // Get the prediction for the canvas data.
  const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const arr = [];
  const pool = [];
  const pooltd = [];
  const data = imgData.data;
  // Grayscale
  for (var i = 0; i < data.length; i += 4) {
    var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
    arr.push(data[i]);
  }
  // Resize(with Average pooling)
  for (var i = 0; i < arr.length; i += 10) {
    var avg = (arr[i] + arr[i + 1] + arr[i + 2] + arr[i + 3] + arr[i + 4] + arr[i + 5] + arr[i + 6] + arr[i + 7] + arr[i + 8] + arr[i + 9]);
    pool.push(avg);
  }
  for (var i = 0; i < pool.length; i += 10) {
    var avg = (pool[i] + pool[i + 28] + pool[i + 28 * 2] + pool[i + 28 * 3] + pool[i + 28 * 4] + pool[i + 28 * 5] + pool[i + 28 * 6] + pool[i + 28 *
      7] + pool[i + 28 * 8] + pool[i + 28 * 9]) / 100;
    pooltd.push(avg);
  }
  // Normalization
  const array = pooltd.map(x=>x/255);

  const input = new onnx.Tensor(new Float32Array(pooltd), "float32", [1, 28, 28, 1]);

  const outputMap = await sess.run([input]);
  // const outputTensor = outputMap.values().next.value;
  const outputTensor = outputMap.get('Identity:0');
  const prediction = outputTensor.data;
  // find max value in array
  var max=-1;
  var idx=0;
  for(var i=0; i<prediction.length; i++){
    if(max<prediction[i]){
      max=prediction[i];
      idx=i;
    }
  }
  console.log(prediction.length);
  document.getElementById("result").innerHTML = fashion[idx];

}

loadingModelPromise.then(() => {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.font = '18px sans-serif';
  ctx.fillText("이미지를 선택해주세요!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
})
