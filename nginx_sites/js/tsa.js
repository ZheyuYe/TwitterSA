/**
 * @Author: Zheyu Ye
 * @Date:   Mon 08 Jul 2019 16:23:20
 * @Last modified by:   Zheyu Ye
 * @Last modified time: Fri 12 Jul 2019 01:49:09
 */

var hosturl = 'http://127.0.0.1:5000/'
// var hosturl = 'http://52.151.71.78:5000/'

window.onload = function visit(){
    update_time();
    add_buttons();
}

$(document).ready(function () {
    $("#start_btn").click(function() {
//       alert( "Handler for .click() called." );
      var update_id = $("#update_id").val();
      // console.log(update_id);
      record_visit(update_id);
      $('#start_rating_banner').removeClass('d-block').addClass('d-none')
      $('#rating_content').removeClass('d-none').addClass('d-block')
      $('#btn_panel').removeClass('d-none').addClass('d-block')
      $('#start_thank').removeClass('d-block').addClass('d-none')

      $('#mask_state').removeClass('d-none').addClass('d-block')
      $('#rating_time').removeClass('d-none').addClass('d-block')
      $('#rating_location').removeClass('d-none').addClass('d-block')


    });
    $("#analysis_form").submit(function() {
        text = $('#analysis_text').val()
        analysis(text)
        return false;
    });

    var isClick = true;
    $("#track_form").submit(function() {
        if(isClick) {
            track_name = $('#track_username').val()
            // console.log(track_name,isClick);
            isClick = false;
            track_user(track_name)
			//定时器
			setTimeout(function() {
				isClick = true;
			}, 2000);//一秒内不能重复点击
		}

        return false;
    });
});
function track_user(track_name){
    var user_id;
    $.ajax({
        type: 'POST',
        url: hosturl+'checkUser',
        dataType: "JSON", // data type expected from server
        data:{
            track_name:track_name,
        },

        success: function (user) {
            // user_id = data.user_id;
            if (user.state=='found'){
                update_profile(user)
                track_activities(user.user_id)
                }
            else{
                noperson()
            }

        },
        error: function(error) {
            console.log('Error:' + error);
            noperson()
        }
    }).done(function() {
});
    function update_profile(user){
        $('#tracking_img_url').attr('src',user.img_url);
        $('#tracking_username').html(user.user_name);
        var screen_name = $('#tracking_screen_name')
        var href = 'http://twitter.com/'+user.screen_name;
        screen_name.html('@'+user.screen_name);
        screen_name.attr("href",href);
        $('#tracking_description').html(user.description);
        $('#tracking_location').html(user.location);
        $('#tracking_followers_count').html(user.followers_count);
        $('#tracking_friends_count').html(user.friends_count);
        $('#tracking_id').val(user.user_id);
    }
    function noperson(){
        removeData()
        $('#tracking_img_url').attr('src','img/noperson.jpg');
        $('#tracking_username').html('No Such Person');
        $('#tracking_description').html('');
        $('#tracking_location').html('');
        $('#tracking_followers_count').html('');
        $('#tracking_friends_count').html('');

    }

    function track_activities(id){
        $.ajax({
            type: 'POST',
            url: hosturl+'track',
            dataType: "JSON", // data type expected from server
            data:{
                id:id
            },
            success: function (data) {
                // sentiment = data.sentiment
                // amount = data.amount
                updateData(data)
            },
            error: function(error) {
                console.log('Error:' + error);
            }
        });
    }
}

function analysis(text){
    $.ajax({
        type: 'POST',
        url: hosturl+'analysis',
        // url:'http://52.151.71.78:5000/analysis',
        dataType: "JSON", // data type expected from server
        data:{
            text:text
        },

        success: function (data) {
            var str = JSON.stringify(data);
            var res = JSON.parse(str);
            $('#analysis_result').removeClass('invisible').addClass('visible')
            $('#analysis_result_pos').css('width',res.pos*100+'%');
            $('#analysis_result_neu').css('width',res.neu*100+'%');
            $('#analysis_result_neg').css('width',res.neg*100+'%');

            $('#analysis_result_pos_sm_text').html(' positive: '+res.pos);
            $('#analysis_result_neu_sm_text').html(' neutral: '+res.neu);
            $('#analysis_result_neg_sm_text').html(' negative: '+res.neg);

            $('#analysis_result_pos_sm').css('width',res.pos*60+'%');
            $('#analysis_result_neu_sm').css('width',res.neu*60+'%');
            $('#analysis_result_neg_sm').css('width',res.neg*60+'%');


            if (res.pos>0.15)
            {
                $('#analysis_result_pos').html('positive:'+res.pos);
            }
            if (res.neu>0.15)
            {
                $('#analysis_result_neu').html('neutral:'+res.neu);
            }
            if (res.neg>0.15)
            {
                $('#analysis_result_neg').html('negative:'+res.neg);
            }
        },
        error: function(error) {
            console.log('Error:' + error);
        }
    });
}


var isClick_button = true;
$("#btn_panel").click(function(event) {

    if(isClick_button) {
        var clicked_id = event.target.id;
        buttons_action(clicked_id);
        isClick_button = false;
        //定时器
        setTimeout(function() {
            isClick_button = true;
        }, 1200);//一秒内不能重复点击
    }
});

// $("#btn_panel").click(function(event) {
//       var clicked_id = event.target.id;
// //       console.log(clicked_id);
//       buttons_action(clicked_id);
// });

// action when clicking rating buttons
function buttons_action(clicked_id){
    var num = $("#rating_count").val();
    var visiting_id = $("#visiting_id").val();
    var status_id = $("#rating_id").val();
    var mask_state = $("#mask_state").val();
    var test_point = $("#test_point").val();
    // console.log("visiting_id "+visiting_id);
    rate_tweet(status_id,clicked_id,visiting_id,mask_state,test_point,num);
    if (num<20){
        get_tweet(status_id);
        // get next tweet after click and rate
    }
    else{
        ending();
    }
}
function ending(){
    // if 25 tweets are all rated, emerging a ending banner showing thanks
    $('#ending_banner').addClass("d-block").removeClass("d-none")
    $('#rating_content').addClass("d-none").removeClass("d-block")
    $('#btn_panel').addClass('invisible').removeClass('visible')


    mask()

}

function mask(){
    $('#rating_time').html('');
    $('#rating_location').html('');
    $('#rating_user_name').html('');
}

function add_buttons(){
    var fullname = new Array("positive","slight positive","neutral","slight negative","negative");
    var pos = document.getElementById('btn_panel')
    var toAdd = document.createDocumentFragment();
    for(var i=1; i < 6; i++){
       var newBtn = document.createElement('a')
       newBtn.innerHTML = fullname[i-1];
       newBtn.id = 'btn_'+i;
       newBtn.className = 'btn btn-outline-primary col-12 col-lg-2';
       // newBtn.setAttribute("onclick","buttons_action(this.id)");
       toAdd.appendChild(newBtn);
    }
    pos.appendChild(toAdd);
    $("#rating_count").val(0);
    $("#rating_id").val(0);
    $("#update_id").val(0);
}

function get_tweet(id){
    var num = $("#rating_count").val();
    $.ajax({
    	type: 'get',
    	url: 'getTweet.php',
    	dataType: "JSON", // data type expected from server
        data:{
            id:id,
            num:num
        },

    	success: function (data) {
            var str = JSON.stringify(data);
            var res = JSON.parse(str);
            // update the content of tweet card including screen_name, time, location, and content
            if (res.state){
                update_card(res);
            }
            else{
                ending();
            }
    	},
    	error: function(error) {
    		console.log('Error:' + error);
    	}
    });
}

function init(update_id){
    var num = $("#rating_count").val();
    $.ajax({
    	type: 'get',
    	url: 'getTweet.php',
    	dataType: "JSON", // data type expected from server
        data:{
            update_id:update_id,
            num:num,
        },

    	success: function (data) {
            var str = JSON.stringify(data);
            var res = JSON.parse(str);
            // update the content of tweet card including screen_name, time, location, and content
            if (res.id){
                update_card(res);
            }
            else{
                ending();
            }
    	},
    	error: function(error) {
    		console.log('Error:' + error);
    	}
    });
}

function update_card(res){
    $("#mask_state").val(res.state);

    var str = res.content;
    // var patt = /(https:\/\/t\.co\/[a-zA-Z\d]*)$/;
    // var result = str.match(patt);
    // if (result){
    //     var url = result[0];
    //     console.log(url);
    // }
    // var body = str.slice(0,result['index']);

    var content = $("#rating_content");
    content.html(str);
    // number increase
    var num = parseInt($("#rating_count").val(),10)+1;
    // output content with number
    $("#rating_count").val(num);
    content.prepend(num+". ");

    if (res.state!=0){
        $("#rating_time").html(res.created_time);

        var screen_name = $('#rating_user_name')
        screen_name.html('@'+res.user_name);
        var href = 'http://twitter.com/'+res.screen_name;
        screen_name.attr("href",href);

        if (res.location!=""){
            var location = res.location;
        }
        else{
            var location = "Unkown Location"
        }
        $("#rating_location").html(location);
    }
    else{
        mask();
    }

    if (res.state!='t'){
        $("#rating_id").val(res.id);
        // $("#test_point").val(-1);
    }
    else{
        $("#test_point").val(res.sentiment);
    }
    // remove url

}
function backLondonTime(){
    var time_zone=1;
    var time = new Date();
    var localTime = time.getTime();
    var localOffset = (time.getTimezoneOffset()+time_zone)*60*1000;
    var LondonTime = new Date(new Date(localTime + localOffset).setHours(0, 0, 0, 0))
    return LondonTime.toString().replace(/[\+\-].*$/,"+0100");
}

function update_time(){
    // var localtime = new Date(new Date().setHours(0, 0, 0, 0));
    $.ajax({
        type: 'GET',
        url: 'checkupdate.php',
        dataType: "JSON", // data type expected from server
        // async: false,//设置为同步传输
        // date:{
            // localtime:localtime,
        // }
        success: function (data) {
            $("#updatetime").html(data.datetime);
            // check the time difference wether more than a day (86400000 ms)
            if (checktime(data.datetime)>86400000){
            // if (checktime(data.datetime)>10){

                console.log("crawl new random Tweets");
                crawl();
            }
            else{
                console.log("no need to crawl");
                // set up value of div that ided as updatetime
                $("#update_id").val(parseInt(data.update_id,10));
                $('#start_btn').removeClass('invisible').addClass('visible')

            }
            // record visit information including time and id of updatetime

        },
        error: function(error) {
            console.log('Error:' + error);
        }
    });
    // check the last update time
    function checktime(lasttime){
        var datetime = new Date(lasttime);
        var localtime = new Date();
        // console.log('updatetime: '+datetime);
        // console.log('localtime: '+localtime);
        var diff = localtime.getTime()-datetime.getTime();
        // console.log(diff);
        return diff
    }

    function crawl(){
        // set the last update as midnight of the day
        // console.log(localtime);
        localtime = backLondonTime()
	    wating_update_id = $.ajax({
            type: 'POST',
            url: hosturl+'crawl',
            dataType: "JSON", // data type expected from server
            data:{
		    localtime:localtime,
	    },
            success: function (data) {
                // get and upload the updatetime
                // str to int
                var intid = parseInt(data.update_id,10);
                $("#update_id").val(intid);
        		if (!isNaN(intid)){
    		        $("#updatetime").html(localtime);
        			}
        		},
            error: function(error) {
                console.log('Error:' + error);
            }
        });

        $.when(wating_update_id).done(function () {
              //要执行的操作
              console.log("show start rating buttons");
              $('#start_btn').removeClass('invisible').addClass('visible')
            });
    }
}

function record_visit(update_id){
    if (isNaN(update_id)){
        console.log("found no update_id");
        return false;
    }
//     console.log(update_id);
    var visiting_time = new Date();
    var time_zone = visiting_time.getTimezoneOffset()/60;
    var visiting_millis  = visiting_time.getTime()-1509191700000;
//console.log(visiting_millis);

    $.ajax({
    type: 'post',
    url: 'record_visit.php',
    dataType: "JSON", // data type expected from server
    data:{
        visiting_time:visiting_time,
        time_zone:time_zone,
        update_id:update_id,
        visiting_millis:visiting_millis,
    },
    success: function (data) {
        var id = parseInt(data,10);
        $("#visiting_id").val(id);
    },
    error: function(error) {
        console.log('Error:' + error);
    }
});
//get first tweet
init(update_id);
}

function rate_tweet(status_id,clicked_id,visiting_id,mask_state,test_point,num){
    var point = parseInt(clicked_id.match(/btn_(\d)/)[1],10);
    // console.log("visiting_id "+visiting_id);
    var clicked_time = new Date().getTime()-1509191700000;

    wating_lastrate = $.ajax({
    	type: 'POST',
    	url: 'rate.php',
    	dataType: "JSON", // data type expected from server
        data:{
            status_id:status_id,
            point:point,
            visiting_id:visiting_id,
            clicked_time:clicked_time,
            mask_state:mask_state,
            test_point:test_point,
            num:num,
        },

    	success: function (data) {
            // var fullname = new Array('Positive','Slightly Positive','neutral','Slightly Negative','Negative');
            // console.log("Tweet id "+status_id + " rated as " + fullname[point-1]);
    	},
    	error: function(error) {
    		console.log('Error:' + error);
    	}
    });

    // $.when(wating_lastrate).done(function () {
    //     var num = $("#rating_count").val();
    //     console.log(num);
    //     if (num == 3){
    //         remove(visiting_id);
    //     }
    // });
}
