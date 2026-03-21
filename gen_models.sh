#!/bin/bash
#生成模型
# RUN <<EOF
# 定义一个生成int8模型的函数
function mv_model(){
    echo "开始移动模型"
    mv $1 $2
    sed -i "s/$1/$2/g" "$2/model_ncnn.py"
    echo "开始移动模型完成"
}
function gen_int8_model() {
    echo "开始生成int8模型"
    cd yolo26n_ncnn_model
    find ../../images/ -type f -name "*.jpeg" -exec realpath {} \; > imagelist.txt
    # ncnn2table model.ncnn.bin model.ncnn.param model.ncnn.table
    # ncnn2table mobilenet-opt.param mobilenet-opt.bin imagelist.txt mobilenet.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
    echo $1
    ncnnoptimize model.ncnn.param model.ncnn.bin model.ncnn.opt.param model.ncnn.opt.bin 0
    ncnn2table model.ncnn.opt.param model.ncnn.opt.bin imagelist.txt model.ncnn.table mean=[0,0,0]  norm=[0.0039, 0.0039, 0.0039] shape=[$1,$1,3] pixel=RGB thread=8 method=eq
    # ./ncnn2table model.ncnn.param test.bin filelist_in0.txt,filelist_in1.txt,filelist_in2.txt test.table shape=[512],[64,1,2],[64,1,2] thread=8 method=kl type=1
    ncnn2int8 model.ncnn.opt.param model.ncnn.opt.bin model.ncnn-int8.param model.ncnn-int8.bin model.ncnn.table
    rm model.ncnn.param model.ncnn.bin
    mv model.ncnn-int8.param model.ncnn.param
    mv model.ncnn-int8.bin model.ncnn.bin
    # rm model.ncnn.opt* imagelist.txt
    cd ..
    # echo "开始生成int8模型完成"
    # gen_int8_model_done=1
}
cd models
yolo export model=yolo26n.pt format=ncnn
mv_model yolo26n_ncnn_model yolo26n_ncnn_model_default
echo 生成模型 320x320
yolo export model=yolo26n.pt format=ncnn imgsz=320
mv_model  yolo26n_ncnn_model yolo26n_ncnn_model_320
echo 生成模型 384x384
yolo export model=yolo26n.pt format=ncnn imgsz=384
mv_model  yolo26n_ncnn_model yolo26n_ncnn_model_384
echo 生成模型 416x416
yolo export model=yolo26n.pt format=ncnn imgsz=416
mv_model  yolo26n_ncnn_model yolo26n_ncnn_model_416
echo 生成模型 320x320 fp16
yolo export model=yolo26n.pt format=ncnn imgsz=320 half=True
mv_model  yolo26n_ncnn_model yolo26n_ncnn_model_320_fp16
echo 生成模型 384x384 fp16
yolo export model=yolo26n.pt format=ncnn imgsz=384 half=True
mv_model  yolo26n_ncnn_model yolo26n_ncnn_model_384_fp16
echo 生成模型 416x416 fp16
yolo export model=yolo26n.pt format=ncnn imgsz=416 half=True
mv_model  yolo26n_ncnn_model yolo26n_ncnn_model_416_fp16
rm -rf yolo26n_ncnn_model_384_int8
echo 生成模型 384x384 int8
yolo export model=yolo26n.pt format=ncnn imgsz=384
gen_int8_model 384
mv_model  yolo26n_ncnn_model yolo26n_ncnn_model_384_int8