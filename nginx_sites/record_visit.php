<?php
include "db_conn.php";

$conn = db_connect();
$visiting_time=$_POST['visiting_time'];
$time_zone=$_POST['time_zone'];
$update_id=$_POST['update_id'];
$visiting_millis=$_POST['visiting_millis'];


// generate a mask sequence where 1 represent shows and 0 represent mask
$ones = '1111111';
$zeros = '0000000';
$tests = 'tttt';
$mask_seq = str_shuffle('01').str_shuffle($zeros.$ones.$tests);

// echo json_encode($mask_seq);

if ($conn){
    try {
        $query = "INSERT INTO `visiting`(visiting_time,time_zone,update_id,visiting_millis,mask_seq) VALUES ('$visiting_time','$time_zone','$update_id','$visiting_millis','$mask_seq')";
        $conn->exec($query);
    }
    catch(PDOException $e)
    {
        echo "fail to insert" . $e->getMessage();
        // return false;
    }
}
// get the lastest recored just insert
$query = "SELECT id FROM `visiting` order by id desc limit 1";
$visiting_id=db_select($conn,$query)['id'];
echo json_encode($visiting_id);

$conn = null;

?>
