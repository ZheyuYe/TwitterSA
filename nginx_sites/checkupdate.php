<?php
include "db_conn.php";

// get the last row at table lastupdate
$conn = db_connect();
// check the table update empty or not
// $query = "SELECT 1 FROM `update`";
// $ifempty=db_select($conn,$query)['1'];
// if (!$ifempty){
    // $localtime=$_POST['localtime'];
    // if table update is empty
// }
$query = "SELECT id,datetime FROM `update` order by id desc limit 1";
$lasttime=db_select($conn,$query);
$array = array('update_id'=>$lasttime['id'],
    'datetime'=>$lasttime['datetime'],
);
echo json_encode($array);
$conn = null;
?>
