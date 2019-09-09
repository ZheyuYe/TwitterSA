<?php
include "db_conn.php";
function rating($conn,$row,$left_millis,$visiting_id){
    $pairs = array(
        1=>'pos',
        2=>'spos',
        3=>'neu',
        4=>'sneg',
        5=>'neg',
        );
    $status_id = $row['status_id'];
    $feeling = $pairs[abs($row['rate_point'])];
    $count = 'count';
    if ($row['rate_point']< 0){
        $feeling=$feeling.'_mask';
        $count = $count.'_mask';
    }
    if ($conn){
        try {
            // update the corresponding column in Status table
            $query = "UPDATE `status` SET $feeling = $feeling +1, $count = $count +1 WHERE id = '$status_id'";
            $conn->exec($query);
            // update the corresponding column in visiting table
            $query = "UPDATE `visiting` SET $feeling = $feeling +1, $count = $count +1 WHERE id = '$visiting_id'";
            $conn->exec($query);

            // get time sapn
            $time_span = $row['rate_millis']-$left_millis;
            // add time diff in visiting table
            $query = "UPDATE `visiting` SET `time_span` = `time_span`+ $time_span WHERE id = '$visiting_id'";
            $conn->exec($query);
            // add time sapn in staus table
            $query = "UPDATE `status` SET `time_span` = `time_span`+ $time_span WHERE id = '$status_id'";
            $conn->exec($query);
        }
        catch(PDOException $e)
        {
            echo $query . "fail <br>" . $e->getMessage();
        }
        // send the feeling back to show in log
    }
}
function temporary($conn,$status_id,$point,$visiting_id,$clicked_time,$mask_state){
    // 0 represent mask
    if ($mask_state==0){
        $point=-$point;
    }
    if ($conn){
        try {
            $query = "INSERT INTO `rating`(status_id,visiting_id,rate_point,rate_millis) VALUES ('$status_id','$visiting_id','$point','$clicked_time')";
            $conn->exec($query);
            // echo $query;
            return 'temporary';
        }
        catch(PDOException $e)
        {
            return "fail to insert" . $e->getMessage();
        }
    }
}
function move($conn,$visiting_id){
    $query = "SELECT visiting_millis FROM `visiting` where id = '$visiting_id'";
    $left_millis=db_select($conn,$query)['visiting_millis'];
    $query = "SELECT status_id,rate_point,rate_millis FROM `rating` where visiting_id = '$visiting_id'";
    $res=$conn->prepare($query);
    $res->execute();
    while($row = $res->fetch(PDO::FETCH_ASSOC)){
        rating($conn,$row,$left_millis,$visiting_id);
        $left_millis = $row['visiting_millis'];
    }
    return 'move';
}

function check_idiot($conn,$visiting_id){
    $maxdiff = 7;
    $query = "SELECT diff FROM `visiting` where id = '$visiting_id'";
    $diff=db_select($conn,$query)['diff'];
    if ($diff<=$maxdiff){
        return FALSE;
    }
    else{
        return TRUE;
    }
}
$conn = db_connect();
$num= $_POST['num'];
// $visiting_id= $_POST['visiting_id'];
$visiting_id= $_POST['visiting_id'];

$test_point = $_POST['test_point'];
$point=$_POST['point'];
$mask_state= $_POST['mask_state'];

if ($mask_state!='t'){
    $status_id = $_POST['status_id'];
    $clicked_time = $_POST['clicked_time'];
    // rating($conn,$status_id,$point,$visiting_id,$clicked_time,$mask_state);
    $state = temporary($conn,$status_id,$point,$visiting_id,$clicked_time,$mask_state);
}
else{
    $point=5-$point;
    $diff = abs($point-$test_point);
    $query = "UPDATE `visiting` SET `diff` = `diff` + $diff WHERE id = '$visiting_id'";
    $conn->exec($query);
    $state = 'verification';
}

if ($num>=20){
    if (check_idiot($conn,$visiting_id)){
        $state = 'idiot';
    }
    else {
        // code...
        $state = move($conn,$visiting_id);
    }
}

    $res = array('state' => $state,);
    echo json_encode($res);
$conn = null;
?>
