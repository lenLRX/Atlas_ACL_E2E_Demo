OLD_PWD=`pwd`
sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
apt update

apt install -y build-essential libpcre3 libpcre3-dev zlib1g-dev openssl libssl-dev
apt install -y cmake git libopencv-dev fonts-droid-fallback libfreetype6-dev libspdlog-dev nlohmann-json-dev python3-dev python3-sklearn python3-numpy python3-opencv python3.7 python3.7-dev

pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pip --upgrade
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install torch torchvision numpy --upgrade

wget --no-check-certificate https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/5.0.5.alpha001/Ascend-cann-toolkit_5.0.5.alpha001_linux-x86_64.run
wget --no-check-certificate https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/Software/AtlasI/A300-3010%201.0.12.2/A300-3010-npu-driver_21.0.3.2_linux-x86_64.run

chmod +x ./A300-3010-npu-driver_21.0.3.2_linux-x86_64.run
./A300-3010-npu-driver_21.0.3.2_linux-x86_64.run --upgrade
chmod +x ./Ascend-cann-toolkit_5.0.5.alpha001_linux-x86_64.run
./Ascend-cann-toolkit_5.0.5.alpha001_linux-x86_64.run --upgrade

git clone https://gitee.com/lenlrx/Atlas_ACL_E2E_Demo.git
cd Atlas_ACL_E2E_Demo
./build.sh

cd $OLD_PWD

while true; do
    read -p "Do you want to reboot now?[Y/N]" yn
    case $yn in
        [Yy]* ) reboot; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
