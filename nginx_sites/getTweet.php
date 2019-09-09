<?php

include 'db_conn.php';
$conn=db_connect();
$update_id=$_GET['update_id'];
$num=$_GET['num'];

$query = "SELECT mask_seq FROM `visiting` order by id desc limit 1;";
$mask_seq = db_select($conn,$query)['mask_seq'];
if ($mask_seq[$num]=='t'){
    $test_id = rand(1,1000);
    $query = "SELECT `text`,`sentiment`,`date`,`user` FROM `sentiment140` where id>= $test_id;";
    $status=db_select($conn,$query);
    
    $array = array(
        'user_name'=>$status['user'],
        'created_time'=>$status['date'],
        'location'=>'',
        'content'=>$status['text'],
        'state'=>$mask_seq[$num],
        'sentiment'=>$status['sentiment'],
    );
}
else{
    if ($update_id){
        $query = "SELECT initial_tweet FROM `update` where id = $update_id;";
        $id=db_select($conn,$query)['initial_tweet'];
        $query = "SELECT id, screen_name, user_name, created_time, location, content from `status` where id = $id;";
    }
    else {
        // get the next tweet
        $id = $_GET['id'];
        $query = "SELECT id, screen_name, user_name, created_time, location, content FROM `status` where id>$id order by id asc limit 1;";
    }

    $status=db_select($conn,$query);
    // echo $content;
    if ($mask_seq[$num]==1){
    $array = array('id'=>$status['id'],
        'screen_name'=>$status['screen_name'],
        'user_name'=>$status['user_name'],
        'created_time'=>$status['created_time'],
        'location'=>$status['location'],
        'content'=>$status['content'],
        'state'=>$mask_seq[$num],
    );
    }
    else{
        $array = array('id'=>$status['id'],
            'content'=>$status['content'],
            'state'=>$mask_seq[$num],
        );
    }
}

echo json_encode($array);
$conn = null;
?>
