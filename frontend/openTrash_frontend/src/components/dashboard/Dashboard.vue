<template>
  <div>
    <div class="button-container">
      <button class="button-segment" @click="handleLeftButtonClick">Vorige Woche</button>
      <div class="button-divider"></div>
      <button class="button-segment" @click="handleMiddleButtonClick">Mittlerer Button</button>
      <div class="button-divider"></div>
      <button class="button-segment" @click="handleRightButtonClick">Nächste Woche</button>
    </div>
    <div class="chart-container">
      <LineChart></LineChart>
    </div>
  </div>
</template>

<script>
import LineChart from './LineChart.vue'

export default {
  name: 'Dashboard',
  components: {
    LineChart
  },
  methods: {
    handleLeftButtonClick() {
      // Logik für den Klick auf den linken Button
    },
    handleMiddleButtonClick() {
      // Logik für den Klick auf den mittleren Button
    },
    handleRightButtonClick() {
      // Logik für den Klick auf den rechten Button
    },
  },
  computed: {
    // Daten abrufen
  }
}
</script>

<style scoped>
.button-container {
  display: flex;
  justify-content: center;
  margin-bottom: 10px;
}

.button-segment {
  flex-grow: 1;
  border-radius: 4px;
  border: none;
  padding: 10px;
}

.button-divider {
  width: 1px;
  background-color: #000;
  margin: 0 10px;
}

.chart-container {
  display: flex;
  justify-content: center;
}

</style>
