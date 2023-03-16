// Importamos TensorFlow.js
import * as tf from '@tensorflow/tfjs';

// Definimos los datos de entrenamiento
const trainingData = tf.tensor2d([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]);

const targetData = tf.tensor2d([
  [0],
  [1],
  [1],
  [0]
]);

// Definimos el modelo
const model = tf.sequential();
model.add(tf.layers.dense({ units: 4, inputShape: [2], activation: 'sigmoid' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Definimos la función de pérdida y el optimizador
const lossFunction = 'meanSquaredError';
const optimizer = tf.train.adam();

// Compilamos el modelo
model.compile({ loss: lossFunction, optimizer: optimizer });

// Entrenamos el modelo
const epochs = 100;
const batchSize = 4;
const history = await model.fit(trainingData, targetData, {
  epochs: epochs,
  batchSize: batchSize
});

// Evaluamos el modelo
const testData = tf.tensor2d([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]);
const predictions = model.predict(testData);
predictions.print();