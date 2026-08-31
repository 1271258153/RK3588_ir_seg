// 加载 rknn 模型并打印所有输入/输出张量属性，用于确认分割模型的 I/O 格式
// 用法: ./rknn_query <model.rknn>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Open %s failed\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(size);
    fread(data, 1, size, fp);
    fclose(fp);
    *model_size = size;
    return data;
}

static const char *type_str(rknn_tensor_type t)
{
    switch (t) {
        case RKNN_TENSOR_FLOAT32: return "FLOAT32";
        case RKNN_TENSOR_FLOAT16: return "FLOAT16";
        case RKNN_TENSOR_INT8:     return "INT8";
        case RKNN_TENSOR_UINT8:    return "UINT8";
        case RKNN_TENSOR_INT16:    return "INT16";
        case RKNN_TENSOR_INT32:    return "INT32";
        case RKNN_TENSOR_INT64:    return "INT64";
        default:                   return "UNKNOWN";
    }
}

static const char *fmt_str(rknn_tensor_format f)
{
    switch (f) {
        case RKNN_TENSOR_NCHW: return "NCHW";
        case RKNN_TENSOR_NHWC: return "NHWC";
        default:               return "UNKNOWN";
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s <rknn model>\n", argv[0]);
        return -1;
    }

    int model_size = 0;
    unsigned char *model_data = load_model(argv[1], &model_size);
    if (!model_data) return -1;

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    if (ret < 0) { printf("rknn_init failed ret=%d\n", ret); return -1; }

    rknn_sdk_version ver;
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    printf("SDK version: %s\n", ver.api_version);

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("n_input=%d  n_output=%d\n\n", io_num.n_input, io_num.n_output);

    for (int i = 0; i < io_num.n_input; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        printf("==== INPUT[%d] ====\n", i);
        printf("  name  : %s\n", attr.name);
        printf("  n_dims: %d  dims=[%d,%d,%d,%d]\n", attr.n_dims,
               attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3]);
        printf("  type  : %s\n", type_str(attr.type));
        printf("  fmt   : %s\n", fmt_str(attr.fmt));
        printf("  fl    : %d\n", attr.fl);
        printf("  scale : %f\n", attr.scale);
        printf("  zp    : %d\n", attr.zp);
        printf("  size  : %d\n", attr.size);
        printf("\n");
    }

    for (int i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        printf("==== OUTPUT[%d] ====\n", i);
        printf("  name  : %s\n", attr.name);
        printf("  n_dims: %d  dims=[%d,%d,%d,%d]\n", attr.n_dims,
               attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3]);
        printf("  type  : %s\n", type_str(attr.type));
        printf("  fmt   : %s\n", fmt_str(attr.fmt));
        printf("  fl    : %d\n", attr.fl);
        printf("  scale : %f\n", attr.scale);
        printf("  zp    : %d\n", attr.zp);
        printf("  size  : %d\n", attr.size);
        printf("\n");
    }

    rknn_destroy(ctx);
    free(model_data);
    return 0;
}
