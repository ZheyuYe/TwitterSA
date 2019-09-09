// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

function number_format(number, decimals, dec_point, thousands_sep) {
  // *     example: number_format(1234.56, 2, ',', ' ');
  // *     return: '1 234,56'
  number = (number + '').replace(',', '').replace(' ', '');
  var n = !isFinite(+number) ? 0 : +number,
    prec = !isFinite(+decimals) ? 0 : Math.abs(decimals),
    sep = (typeof thousands_sep === 'undefined') ? ',' : thousands_sep,
    dec = (typeof dec_point === 'undefined') ? '.' : dec_point,
    s = '',
    toFixedFix = function(n, prec) {
      var k = Math.pow(10, prec);
      return '' + Math.round(n * k) / k;
    };
  // Fix for IE parseFloat(0.55).toFixed(0) = 0;
  s = (prec ? toFixedFix(n, prec) : '' + Math.round(n)).split('.');
  if (s[0].length > 3) {
    s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep);
  }
  if ((s[1] || '').length < prec) {
    s[1] = s[1] || '';
    s[1] += new Array(prec - s[1].length + 1).join('0');
  }
  return s.join(dec);
}

// Area Chart Example
var ctx = $('#myAreaChart')
var curDate = new Date();
var options = { weekday: 'short', month: 'short', day: '2-digit'};
var labels = new Array();
var i;
for (i = 6; i > -1 ; i--) {
  day = new Date(curDate.getTime() - 86400000*i);
  labels[6-i]=day.toLocaleDateString('en-GB',options);
}

var myLineChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: labels,
    datasets: [
        {
          label: "Sentiment",
          yAxisID: 'y-axis-S',
          lineTension: 0.3,
          fill: false,
          backgroundColor: "rgba(39, 117, 182, 0.05)",
          borderColor: "rgba(39, 117, 182, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(39, 117, 182, 1)",
          pointBorderColor: "rgba(39, 117, 182, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(39, 117, 182, 1)",
          pointHoverBorderColor: "rgba(39, 117, 182, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          // data: [0,0,0,0,0,0,0],
          // data: [NaN,NaN,NaN,NaN,NaN,NaN,NaN],
          data:[],

      },
      {
        label: "Amount",
        yAxisID: 'y-axis-A',
        lineTension: 0.3,
        fill: false,
        backgroundColor: "rgba(255,156,65, 0.05)",
        borderColor: "rgba(255,156,65, 1)",
        pointRadius: 3,
        pointBackgroundColor: "rgba(255,156,65, 1)",
        pointBorderColor: "rgba(255,156,65, 1)",
        pointHoverRadius: 3,
        pointHoverBackgroundColor: "rgba(255,156,65, 1)",
        pointHoverBorderColor: "rgba(255,156,65, 1)",
        pointHitRadius: 10,
        pointBorderWidth: 2,
        // data: [NaN,NaN,NaN,NaN,NaN,NaN,NaN],
        data:[],
    },

],
  },
  options: {
    maintainAspectRatio: false,
    layout: {
      padding: {
        left: 5,
        right: 5,
        top: 0,
        bottom: 0
      }
    },
    scales: {
      xAxes: [{
        time: {
          unit: 'date'
        },
        gridLines: {
          display: false,
          drawBorder: false
        },
        ticks: {
          maxTicksLimit: 7
        }
      }],
      yAxes: [
          {
              id: 'y-axis-S',
              type: 'linear',
              position: 'left',

              ticks: {
                  maxTicksLimit: 3,
                  // max:1.1,
                  // min:-1.1,
                  // suggestedMax:1.1,
                  stepSize: 1,
                  // padding: 5,
                  // autoSkip: true,

                  callback: function(value, index, values) {
                    return value;
              }
            },
            gridLines: {
                color: "rgb(234, 236, 244)",
                  zeroLineColor: "rgb(234, 236, 244)",
                  drawBorder: true,
                  borderDash: [2],
                  zeroLineBorderDash: [2]
            }
        },
        {
            id: 'y-axis-A',
            type: 'linear',
            position: 'right',
            ticks:{
                min:0,
                maxTicksLimit: 5,
            },
            // gridLines: {
                // display:false,
                // drawBorder: false,
            // }
        }
    ]},

    legend: {
      display: true
    },
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      // backgroundColor: 'rgba(0,0,0,0.8)',
      bodyFontColor: "#858796",
      titleMarginBottom: 10,
      titleFontColor: '#6e707e',
      titleFontSize: 14,
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 5,
      yPadding: 5,
      displayColors: true,
      intersect: false,
      mode: 'index',
      caretPadding: 2,
      callbacks: {
        label: function(tooltipItem, chart) {
          var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
          return datasetLabel + ': '+tooltipItem.yLabel;
        }
      }
  },
  annotation: {
        annotations: [{
            type: 'line',
            mode: 'horizontal',
            scaleID: 'y-axis-S',
            value: '1',
            borderColor: 'red',
            borderWidth: 3,
            // fontSize: 20,
            label: {
                content:"Most Positive",
                backgroundColor: 'rgba(255,255,255,0.8)',
                fontColor: "#000",
                enabled: true,
                fontStyle: "slight",
                position: "center",
                yAdjust: 5,
            }
        },{
            type: 'line',
            mode: 'horizontal',
            scaleID: 'y-axis-S',
            value: '-1',
            borderColor: 'red',
            borderWidth: 3,
            label: {
                content:"Most Negative",
                backgroundColor: 'rgba(255,255,255,0.8)',
                fontSize: 16,
                fontColor: "#000",
                enabled: true,
                fontStyle: "slight",
                position: "center",
                yAdjust: -5,
            },
        },{
            type: 'line',
            mode: 'horizontal',
            scaleID: 'y-axis-S',
            value: '0',
            borderColor: 'red',
            borderWidth: 2,
            label: {
                content:"Neutral",
                backgroundColor: 'rgba(255,255,255,0.8)',
                fontSize: 16,
                fontColor: "#000",
                enabled: true,
                fontStyle: "slight",
                position: "center",
            }
        }],
        drawTime: "beforeDatasetsDraw" // (default)
    },
  }
});

function updateData(trackData) {
    myLineChart.data.datasets[0].data = trackData.sentiment
    myLineChart.data.datasets[1].data = trackData.amount
    myLineChart.update();
}

function removeData() {
    myLineChart.data.datasets[0].data = []
    myLineChart.data.datasets[1].data = []
    myLineChart.update();
}

// $('#myAreaChart').click(function(e) {
//     var activeBars = myLineChart.getElementAtEvent(e);
//     console.log(activeBars);
//     console.log('hello');
// }
$("#myAreaChart").click(
    function(evt){
        var activePoints = myLineChart.getElementAtEvent(evt)[0];
        var index = activePoints._index;
        if (index>=0 & index <7){
            day = new Date(curDate.getTime() - 86400000*(6-index));
            daystring=day.toLocaleDateString('en-GB',{ weekday: 'long', month: 'long', day: '2-digit'});
            $("#details_day").html(daystring);
            get_details(6-index);
            $("#track_modal").modal();
        }
    }
);

function get_details(index){
    user_id = $("#tracking_id").val()
    // console.log(user_id);
    $.ajax({
        type: 'POST',
        url: hosturl+'details',
        dataType: "JSON", // data type expected from server
        data:{
        index:index,
        user_id:user_id,
    },
        success: function (data) {
                show_details(data)
            },
        error: function(error) {
            console.log('Error:' + error);
        }
    });
    function show_details(data){
        var pos = document.getElementById('details_body')
        pos.innerHTML = '';
        var toAdd = document.createDocumentFragment();

        for(var i=0; i < data.length; i++){
           var newTweet = document.createElement('div')
           newTweet.className = 'row details_tweet';

           var newTime = document.createElement('h7')
           newTime.className= 'col-5 details_time';
           newTime.innerHTML = data[i][0];
           var newContent = document.createElement('h7')
           newContent.className= 'col-12 details_content';
           newContent.innerHTML = data[i][1];
           var newSentiment = document.createElement('h7')
           newSentiment.className= 'col-7 details_sentiment';
           newSentiment.innerHTML = data[i][2];
           if (data[i][2]>= 0.05){
               newSentiment.innerHTML += ' 😊';
           }
           else if (data[i][2]<= -0.05){
               newSentiment.innerHTML += ' 😞';
           }
           else{
               newSentiment.innerHTML += ' 😐';
           }

           newTweet.appendChild(newTime);
           newTweet.appendChild(newSentiment);
           newTweet.appendChild(newContent);

           toAdd.appendChild(newTweet);
        }
        pos.appendChild(toAdd);
    }
}
