    <script>
    export default {
		name: 'LineChart',
      data() {
        return {
          chart: null,
          options: {
            animationEnabled: true,
            exportEnabled: true,
            title:{
              text: "Classified correctly"
            },
            axisX: {
              title: "Time ⟶"
            },
            axisY: {
              title: "Hours ⟶"
            },
            data: [{
              type: "spline",
              yValueFormatString: "#,###.## hours",
              dataPoints: [
                { label: "Mon", y: 10 },
                { label: "Tue", y: 11 },
                { label: "Wed", y: 12 },
                { label: "Thur", y: 13 },
                { label: "Fri", y: 14 },
                { label: "Sat", y: 14.5 },
                { label: "Sun", y: 14 },
                { label: "Aug", y: 13.5 },
                { label: "Sep", y: 12.5 },
                { label: "Oct", y: 11.5 },
                { label: "Nov", y: 10.5 },
                { label: "Dec", y: 10 }
              ]
            }]
          },
          styleOptions: {
            width: "460px",
            height: "360px"
          }
        }
      },
      methods: {
        chartInstance(chart) {
          this.chart = chart;
        }
      }
    }
    </script>
    <template>
        <CanvasJSChart :options="options" :style="styleOptions" @chart-ref="chartInstance"/>
    </template>                              