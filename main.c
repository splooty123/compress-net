#define _CRT_SECURE_NO_WARNINGS
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <direct.h>
#include <windows.h>
#define PATH_SEP '\\'
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#define PATH_SEP '/'
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "neural_net.h"
#include "stb_image.h"

#define MAX_EPOCH 1001
#define APP_PATH_MAX 4096

static unsigned char* load_image(const char* path, int newW, int newH);

static inline void setup_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    if (hOut != INVALID_HANDLE_VALUE && GetConsoleMode(hOut, &dwMode)) {
        SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#endif
}

static void clear_console(void) {
    printf("\033[2J\033[H");
    fflush(stdout);
}

static char* get_current_directory(char* buffer, size_t size) {
#ifdef _WIN32
    return _getcwd(buffer, (int)size);
#else
    return getcwd(buffer, size);
#endif
}

static int ensure_directory(const char* path) {
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) {
        return 1;
    }
#else
    if (mkdir(path, 0755) == 0 || errno == EEXIST) {
        return 1;
    }
#endif
    printf("Failed to create directory: %s\n", path);
    return 0;
}

static int path_join(char* dst, size_t dst_size, const char* left, const char* right) {
    size_t left_len = strlen(left);
    char sep_buf[2] = { PATH_SEP, '\0' };
    const char* sep = "";

    if (left_len > 0 && left[left_len - 1] != '/' && left[left_len - 1] != '\\') {
        sep = sep_buf;
    }

    int written = snprintf(dst, dst_size, "%s%s%s", left, sep, right);
    if (written < 0 || (size_t)written >= dst_size) {
        printf("Path too long: %s/%s\n", left, right);
        return 0;
    }

    return 1;
}

static int is_supported_image_file(const char* path) {
    int width, height, channels;
    return stbi_info(path, &width, &height, &channels);
}

static int append_loaded_image(const char* fullpath, unsigned char*** imgs, int* img_count, int width, int height) {
    printf("Loading: %d %s\n", *img_count, fullpath);

    unsigned char* img = load_image(fullpath, width, height);
    if (!img) {
        printf("Skipping bad image: %s\n", fullpath);
        return 1;
    }

    unsigned char** new_imgs = realloc(*imgs, sizeof(unsigned char*) * ((size_t)*img_count + 1));
    if (!new_imgs) {
        printf("Memory allocation failed.\n");
        free(img);
        return 0;
    }

    *imgs = new_imgs;
    (*imgs)[(*img_count)++] = img;
    return 1;
}

void makeBMP(const char* filename, int width, int height, const unsigned char* rgb) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    int rowSize = (3 * width + 3) & ~3;
    int dataSize = rowSize * height;
    int fileSize = 54 + dataSize;

    unsigned char header[54] = {
        'B','M',
        0,0,0,0,    // file size
        0,0, 0,0,
        54,0,0,0,   // pixel data offset
        40,0,0,0,   // DIB header size
        0,0,0,0,    // width
        0,0,0,0,    // height
        1,0,        // planes
        24,0,       // bpp
        0,0,0,0,    // compression
        0,0,0,0,    // image size
        0x13,0x0B,0,0,
        0x13,0x0B,0,0,
        0,0,0,0,
        0,0,0,0
    };

    header[2] = fileSize;
    header[3] = fileSize >> 8;
    header[4] = fileSize >> 16;
    header[5] = fileSize >> 24;

    header[18] = width;
    header[19] = width >> 8;
    header[20] = width >> 16;
    header[21] = width >> 24;

    header[22] = height;
    header[23] = height >> 8;
    header[24] = height >> 16;
    header[25] = height >> 24;

    fwrite(header, 1, 54, f);

    unsigned char* row = (unsigned char*)malloc(rowSize);

    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int i = (y * width + x) * 3;
            row[x * 3] = rgb[i + 2];
            row[x * 3 + 1] = rgb[i + 1];
            row[x * 3 + 2] = rgb[i];
        }
        for (int p = width * 3; p < rowSize; p++) row[p] = 0;
        fwrite(row, 1, rowSize, f);
    }

    free(row);
    fclose(f);
}

static unsigned char* load_image(const char* path, int newW, int newH) {
    int w, h, channels;
    unsigned char* input = stbi_load(path, &w, &h, &channels, 3);

    if (!input) {
        printf("Failed to load image: %s\n", path);
        return NULL;
    }

    unsigned char* output = malloc(newW * newH * 3);
    if (!output) {
        printf("Out of memory resizing: %s\n", path);
        stbi_image_free(input);
        return NULL;
    }

    for (int y = 0; y < newH; y++) {
        for (int x = 0; x < newW; x++) {
            int srcX = x * w / newW;
            int srcY = y * h / newH;
            int srcIndex = (srcY * w + srcX) * 3;
            int dstIndex = (y * newW + x) * 3;

            output[dstIndex + 0] = input[srcIndex + 0];
            output[dstIndex + 1] = input[srcIndex + 1];
            output[dstIndex + 2] = input[srcIndex + 2];
        }
    }

    stbi_image_free(input);
    return output;
}

static int load_all_images(const char* folder, unsigned char*** out_imgs, int* count, int width, int height) {
    *out_imgs = NULL;
    *count = 0;

    char cwd[APP_PATH_MAX];
    if (get_current_directory(cwd, sizeof(cwd))) {
        printf("CWD = %s\n", cwd);
    }

    printf("Searching for stb_image-supported images in: %s\n", folder);

    unsigned char** imgs = NULL;
    int img_count = 0;

#ifdef _WIN32
    char search_path[APP_PATH_MAX];
    if (!path_join(search_path, sizeof(search_path), folder, "*")) {
        return 0;
    }

    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(search_path, &fd);

    if (h == INVALID_HANDLE_VALUE) {
        printf("Could not open image folder: %s\n", folder);
        return 0;
    }

    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            char fullpath[APP_PATH_MAX];
            if (path_join(fullpath, sizeof(fullpath), folder, fd.cFileName) && is_supported_image_file(fullpath)) {
                if (!append_loaded_image(fullpath, &imgs, &img_count, width, height)) {
                    FindClose(h);
                    return 0;
                }
            }
        }
    } while (FindNextFileA(h, &fd));

    FindClose(h);
#else
    DIR* dir = opendir(folder);
    if (!dir) {
        printf("Could not open image folder: %s\n", folder);
        return 0;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        char fullpath[APP_PATH_MAX];
        if (!path_join(fullpath, sizeof(fullpath), folder, entry->d_name)) {
            continue;
        }

        struct stat st;
        if (stat(fullpath, &st) == 0 && S_ISDIR(st.st_mode)) {
            continue;
        }

        if (!is_supported_image_file(fullpath)) {
            continue;
        }

        if (!append_loaded_image(fullpath, &imgs, &img_count, width, height)) {
            closedir(dir);
            return 0;
        }
    }

    closedir(dir);
#endif

    printf("Loaded %d valid images.\n", img_count);

    *out_imgs = imgs;
    *count = img_count;
    if (img_count == 0) {
        printf("No supported image files found in folder: %s\n", folder);
    }

    return img_count > 0;
}

static void draw_image(const unsigned char* img, int w, int h, int width, int height) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int srcX = x * width / w;
            int srcY = y * height / h;
            int idx = (srcY * width + srcX) * 3;

            int r = img[idx + 0];
            int g = img[idx + 1];
            int b = img[idx + 2];
            printf("\033[38;2;%d;%d;%dm██", r, g, b);
        }
        printf("\033[0m\n");
    }
}

static float* normalize_rgb(unsigned char* img, unsigned int size) {
    float* img_float = (float*)malloc(size * sizeof(float));
    if (!img_float)exit(1);
    for (unsigned int i = 0; i < size; i++) {
        img_float[i] = img[i] / 255.0f;
    }
    return img_float;
}

static unsigned char* unnormalize_rgb(const float* img_float, unsigned int size) {
    unsigned char* out = (unsigned char*)malloc(size * sizeof(unsigned char));
    if (!out) exit(1);
    for (unsigned int i = 0; i < size; i++) {
        float v = img_float[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        v = v * 255.0f;
        out[i] = (unsigned char)(v + 0.5f);
    }

    return out;
}

static inline void progress_bar(int a, int b, int length) {
    printf("[");
    for (int i = 0; i < length; i++) {
        if ((i + 0.5f) * (float)a / (float)length < b) {
            printf("█");
        }
        else {
            printf(":");
        }
    }
    printf("]");
}

static void shuffle_indices(int* indices, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

int main() {
    srand((unsigned int)time(NULL));

    int draw_pictures = 0;
    int output_images = 0;
    unsigned int width = 0;
    unsigned int height = 0;

    printf("image output: ");
    scanf(" %d", &output_images);

    printf("draw to console: ");
    scanf(" %d", &draw_pictures);

    printf("width: ");
    scanf(" %u", &width);

    printf("height: ");
    scanf(" %u", &height);

    neural_net betavae;
    unsigned int structure[] = { width * height * 3, 256, 128, 256, width * height * 3 };
    neural_net_init(&betavae, 5, structure);
    setup_console();

    if (output_images && !ensure_directory("image_output")) {
        neural_net_free(&betavae);
        return 1;
    }

    unsigned char** imgs = NULL;
    int img_count = 0;

    if (!load_all_images("images", &imgs, &img_count, width, height)) {
        neural_net_free(&betavae);
        return 1;
    }
    
    float** imgs_float = malloc(sizeof(float*) * img_count);

    if (!imgs_float)exit(1);
    for (int i = 0; i < img_count; i++) {
        imgs_float[i] = normalize_rgb(imgs[i], width * height * 3);
    }

    int* order = (int*)malloc(sizeof(int) * (size_t)img_count);
    if (!order)exit(1);
    for (int i = 0; i < img_count; i++) {
        order[i] = i;
    }

    printf("loaded %d images", img_count);

    clear_console();
    const float base_lr = 0.001f;
    float losses[MAX_EPOCH];
    for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {
        float error = 0;
        float lr = base_lr / (1.0f + 0.001f * (float)epoch);
        shuffle_indices(order, img_count);

        for (int k = 0; k < img_count; k++) {
            int i = order[k];
            forward_prop(&betavae, imgs_float[i]);
            error += backward_prop(&betavae, imgs_float[i], lr) / img_count;
        }
        losses[epoch] = error;
        
        if (output_images || draw_pictures) {
            int draw_image_index = 1;

            forward_prop(&betavae, imgs_float[draw_image_index]);
            float* layer = get_layer(&betavae, betavae.size - 1);
            unsigned char* draw_img = unnormalize_rgb(layer, width * height * 3);

            if ((epoch % 10 == 0) && output_images) {
                unsigned char* bmp = (unsigned char*)malloc(width * height * 6 * sizeof(unsigned char));
                if (!bmp)exit(1);
                memcpy(bmp, imgs[draw_image_index], width * height * 3);
                memcpy(bmp + width * height * 3, draw_img, width * height * 3);
                char filename[16];
                char path[APP_PATH_MAX];
                snprintf(filename, sizeof(filename), "%04d.bmp", epoch);
                if (path_join(path, sizeof(path), "image_output", filename)) {
                    makeBMP(path, width, 2 * height, bmp);
                }
                free(bmp);
            }

            printf("\033[H");

            if (draw_pictures) {
                draw_image(imgs[draw_image_index], width, height, width, height);
                draw_image(draw_img, width, height, width, height);
            }

            printf("\n");
            progress_bar(MAX_EPOCH, epoch, 20);
            printf(" epoch: %d loss: %f", epoch + 1, error);

            free(layer);
            free(draw_img);
        } else {
            printf("\033[H");
            progress_bar(MAX_EPOCH, epoch, 20);
            printf(" epoch: %d loss: %f", epoch + 1, error);
        }
    }

    for (int i = 0; i < img_count; i++) {
        free(imgs[i]);
        free(imgs_float[i]);
    }

    clear_console();
    for (int i = 0; i < MAX_EPOCH; i++) {
        printf("%f, ", losses[i]);
    }

    free(imgs);
    free(imgs_float);
    free(order);
    neural_net_free(&betavae);

    return 0;
}
