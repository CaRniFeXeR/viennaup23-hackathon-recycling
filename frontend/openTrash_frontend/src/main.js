import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'

import CanvasJSChart from '@canvasjs/vue-charts';

const app = createApp(App);
app.use(CanvasJSChart); // install the CanvasJS Vuejs Chart Plugin
app.mount('#app')