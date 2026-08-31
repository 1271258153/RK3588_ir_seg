set -e

# TARGET_SOC="rk3588"
GCC_COMPILER=/usr/local/arm64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu

export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

if [ ! -d "${BUILD_DIR}" ]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux
make -j8
make install
cd -

# 运行示例 (摄像头 /dev/video11 已在 main.cc 中硬编码，第二个参数实际未使用)
# cd install/pidnet_Linux/ && ./pidnet ./model/RK3588/best.rknn dummy.jpg
# 使用 yolov5s 模型
# cd install/pidnet_Linux/ && ./pidnet ./model/RK3588/yolov5s-640-640.rknn dummy.jpg

