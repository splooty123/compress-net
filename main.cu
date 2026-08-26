#define _CRT_SECURE_NO_WARNINGS
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define WIN32_LEAN_AND_MEAN
#define NOGDI
#define NOUSER
#define NOMINMAX
#include <windows.h>
#include <direct.h>
#define PATH_SEP '\\'

#include "raylib.h"
#include "neural_net.cuh"

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
#define MAX_EPOCH    2000
#define TRAIN_BATCH  128u
#define BASE_LR      0.0005f
#define APP_PATH_MAX 4096

#define WIN_W   512

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------
static char* get_cwd(char* buf, size_t n) {
    return _getcwd(buf, (int)n);
}

static int ensure_dir(const char* path) {
    if (_mkdir(path) == 0 || errno == EEXIST) return 1;
    fprintf(stderr, "Failed to create directory: %s\n", path);
    return 0;
}

static int path_join(char* dst, size_t dst_size, const char* left, const char* right) {
    size_t ll = strlen(left);
    char sep[2] = { PATH_SEP, '\0' };
    const char* s = (ll > 0 && left[ll - 1] != '/' && left[ll - 1] != '\\') ? sep : "";
    int n = snprintf(dst, dst_size, "%s%s%s", left, s, right);
    if (n < 0 || (size_t)n >= dst_size) {
        fprintf(stderr, "Path too long: %s + %s\n", left, right);
        return 0;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Image I/O
// ---------------------------------------------------------------------------
static unsigned char* load_image(const char* path, int newW, int newH) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }

    unsigned char hdr[54];
    if (fread(hdr, 1, 54, f) != 54 || hdr[0] != 'B' || hdr[1] != 'M') {
        fprintf(stderr, "Not a BMP: %s\n", path); fclose(f); return NULL;
    }

    int w = hdr[18] | hdr[19] << 8 | hdr[20] << 16 | hdr[21] << 24;
    int h = hdr[22] | hdr[23] << 8 | hdr[24] << 16 | hdr[25] << 24;
    int bpp = hdr[28] | hdr[29] << 8;
    int offset = hdr[10] | hdr[11] << 8 | hdr[12] << 16 | hdr[13] << 24;

    if (bpp != 24) {
        fprintf(stderr, "Only 24-bit BMP supported: %s\n", path);
        fclose(f); return NULL;
    }

    fseek(f, offset, SEEK_SET);

    int row_sz = (3 * w + 3) & ~3;
    unsigned char* src = (unsigned char*)malloc((size_t)row_sz * abs(h));
    unsigned char* dst = (unsigned char*)malloc((size_t)newW * newH * 3);
    if (!src || !dst) { free(src); free(dst); fclose(f); return NULL; }

    fread(src, 1, (size_t)row_sz * abs(h), f);
    fclose(f);

    for (int y = 0; y < newH; y++) {
        for (int x = 0; x < newW; x++) {
            int sx = x * w / newW;
            int sy = y * h / newH;
            int bmp_y = (h - 1) - sy;
            const unsigned char* p = src + bmp_y * row_sz + sx * 3;
            unsigned char* q = dst + (y * newW + x) * 3;
            q[0] = p[2];
            q[1] = p[1];
            q[2] = p[0];
        }
    }

    free(src);
    return dst;
}

static int load_images(const char* folder,
    unsigned char*** out_imgs,
    int width, int height)
{
    system("python bmp.py");

    *out_imgs = NULL;
    int count = 0;

    char cwd[APP_PATH_MAX];
    if (get_cwd(cwd, sizeof(cwd))) printf("CWD: %s\n", cwd);
    printf("Loading images from: %s\n", folder);

    char pattern[APP_PATH_MAX];
    if (!path_join(pattern, sizeof(pattern), folder, "*")) return 0;

    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open folder: %s\n", folder); return 0;
    }
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

        char fullpath[APP_PATH_MAX];
        if (!path_join(fullpath, sizeof(fullpath), folder, fd.cFileName)) continue;

        size_t nl = strlen(fd.cFileName);
        if (nl < 4 || _stricmp(fd.cFileName + nl - 4, ".bmp") != 0) continue;

        unsigned char* img = load_image(fullpath, width, height);
        if (!img) continue;

        unsigned char** tmp = (unsigned char**)realloc(*out_imgs, sizeof(unsigned char*) * (size_t)(count + 1));
        if (!tmp) { free(img); FindClose(h); return count; }
        *out_imgs = tmp;
        (*out_imgs)[count++] = img;
        printf("  [%d] %s\n", count - 1, fd.cFileName);

    } while (FindNextFileA(h, &fd));
    FindClose(h);

    printf("Loaded %d images.\n", count);
    return count;
}

// Write a 24-bit BMP (bottom-up, BGR row-padded to 4 bytes).
static void write_bmp(const char* path, int width, int height, const unsigned char* rgb) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write BMP: %s\n", path); return; }

    int row_sz = (3 * width + 3) & ~3;
    int data_sz = row_sz * height;
    int file_sz = 54 + data_sz;

    unsigned char hdr[54] = {
        'B','M',
        (unsigned char)file_sz,        (unsigned char)(file_sz >> 8),
        (unsigned char)(file_sz >> 16),(unsigned char)(file_sz >> 24),
        0,0,0,0,
        54,0,0,0,
        40,0,0,0,
        (unsigned char)width,         (unsigned char)(width >> 8),
        (unsigned char)(width >> 16), (unsigned char)(width >> 24),
        (unsigned char)height,        (unsigned char)(height >> 8),
        (unsigned char)(height >> 16),(unsigned char)(height >> 24),
        1,0, 24,0,
        0,0,0,0, 0,0,0,0,
        0x13,0x0B,0,0, 0x13,0x0B,0,0,
        0,0,0,0, 0,0,0,0
    };
    fwrite(hdr, 1, 54, f);

    unsigned char* row = (unsigned char*)malloc((size_t)row_sz);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            const unsigned char* p = rgb + (y * width + x) * 3;
            row[x * 3 + 0] = p[2];
            row[x * 3 + 1] = p[1];
            row[x * 3 + 2] = p[0];
        }
        memset(row + width * 3, 0, (size_t)(row_sz - width * 3));
        fwrite(row, 1, (size_t)row_sz, f);
    }
    free(row);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Float <-> byte conversion
// ---------------------------------------------------------------------------
static float* img_to_float(const unsigned char* img, unsigned int n) {
    float* f = (float*)malloc(n * sizeof(float));
    if (!f) { fprintf(stderr, "OOM img_to_float\n"); exit(1); }
    for (unsigned int i = 0; i < n; i++) f[i] = img[i] / 255.0f;
    return f;
}

static unsigned char* float_to_img(const float* f, unsigned int n) {
    unsigned char* out = (unsigned char*)malloc(n);
    if (!out) { fprintf(stderr, "OOM float_to_img\n"); exit(1); }
    for (unsigned int i = 0; i < n; i++) {
        float v = f[i] < 0.0f ? 0.0f : f[i] > 1.0f ? 1.0f : f[i];
        out[i] = (unsigned char)(v * 255.0f + 0.5f);
    }
    return out;
}

static void progress_bar(int total, int done, int len) {
    printf("[");
    for (int i = 0; i < len; i++)
        printf((i + 0.5f) * total / len < done ? "\xe2\x96\x88" : ":");
    printf("]");
}

// ---------------------------------------------------------------------------
// Training helpers
// ---------------------------------------------------------------------------
static void shuffle(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

void init_decoder(neural_net* decoder, neural_net* full) {
    int dec_layers = full->size - (full->size / 2);
    decoder->size = dec_layers;
    decoder->structure = (unsigned int*)malloc(dec_layers * sizeof(unsigned int));
    for (int i = 0; i < dec_layers; i++)
        decoder->structure[i] = full->structure[full->size / 2 + i];
    neural_net_init(decoder, dec_layers, decoder->structure, 1);
}

void copy_decoder_weights(neural_net* net, neural_net* dec) {
    int enc_layers = net->size / 2;
    int w_offset = 0;
    int b_offset = 0;
    int dec_w_offset = 0;
    int dec_b_offset = 0;

    for (int i = 0; i < net->size - 1; i++) {
        int in = net->structure[i];
        int out = net->structure[i + 1];
        int w_size = in * out;
        int b_size = out;

        if (i < enc_layers) {
            w_offset += w_size;
            b_offset += b_size;
            continue;
        }

        CUDA_CHECK(cudaMemcpy(
            dec->weights + dec_w_offset,
            net->weights + w_offset,
            w_size * sizeof(float),
            cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaMemcpy(
            dec->bias + dec_b_offset,
            net->bias + b_offset,
            b_size * sizeof(float),
            cudaMemcpyDeviceToDevice));

        w_offset += w_size;
        b_offset += b_size;
        dec_w_offset += w_size;
        dec_b_offset += b_size;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(void) {
    srand((unsigned int)time(NULL));

    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        GetConsoleMode(hOut, &mode);
        mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, mode);
    }

    // --- User input ---
    int          output_images = 0;
    int          draw_pictures = 1;
    unsigned int width = 0;
    unsigned int height = 0;

    printf("Save BMP output (0/1): "); scanf("%d", &output_images);
    printf("Image width:  ");          scanf("%u", &width);
    printf("Image height: ");          scanf("%u", &height);

    if (width == 0 || height == 0) {
        fprintf(stderr, "Error: width and height must be > 0 (got %u x %u)\n", width, height);
        return 1;
    }

    if (output_images && !ensure_dir("image_output")) return 1;

    // --- Load images ---
    unsigned char** imgs = NULL;
    int img_count = load_images("images", &imgs, (int)width, (int)height);
    if (img_count == 0) { fprintf(stderr, "No images found.\n"); return 1; }

    unsigned int pixel_floats = width * height * 3;

    float** imgs_f = (float**)malloc(sizeof(float*) * (size_t)img_count);
    if (!imgs_f) { fprintf(stderr, "OOM\n"); return 1; }
    for (int i = 0; i < img_count; i++)
        imgs_f[i] = img_to_float(imgs[i], pixel_floats);

    float** d_imgs = (float**)malloc(sizeof(float*) * (size_t)img_count);
    if (!d_imgs) { fprintf(stderr, "OOM\n"); return 1; }
    for (int i = 0; i < img_count; i++) {
        CUDA_CHECK(cudaMalloc((void**)&d_imgs[i], pixel_floats * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_imgs[i], imgs_f[i],
            pixel_floats * sizeof(float), cudaMemcpyHostToDevice));
    }

    int* order = (int*)malloc(sizeof(int) * (size_t)img_count);
    if (!order) { fprintf(stderr, "OOM\n"); return 1; }
    for (int i = 0; i < img_count; i++) order[i] = i;

    // --- Network ---
    gpu_init();

    unsigned int batch = (img_count >= (int)TRAIN_BATCH) ? TRAIN_BATCH : (unsigned int)img_count;
    unsigned int structure[] = { pixel_floats, 512, 256, 512, pixel_floats };
    neural_net   net;
    neural_net_init(&net, 5, structure, batch);

    float* h_batch_in = (float*)malloc((size_t)batch * pixel_floats * sizeof(float));
    if (!h_batch_in) { fprintf(stderr, "OOM h_batch_in\n"); return 1; }

    float* d_targets = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_targets, (size_t)batch * pixel_floats * sizeof(float)));

    printf("\nTraining %d epochs, batch=%u, lr=%.4f\n\n", MAX_EPOCH, batch, BASE_LR);

    float* losses = (float*)malloc(MAX_EPOCH * sizeof(float));
    if (!losses) { fprintf(stderr, "OOM\n"); return 1; }

    // --- Window (always open — needed for post-training UI) ---
    int train_win_w = (int)width * (WIN_W / (int)width);
    int train_win_h = (int)height * 2 * (WIN_W / (int)width);
    InitWindow(train_win_w, train_win_h, "CompressNet");
    SetTargetFPS(60);

    // Texture covers original (top half) + reconstruction (bottom half)
    unsigned char* frame_buf = (unsigned char*)malloc(width * height * 2 * 4);
    memset(frame_buf, 0, width * height * 2 * 4);
    Image blank = GenImageColor((int)width, (int)height * 2, BLACK);
    Texture2D screenTex = LoadTextureFromImage(blank);
    UnloadImage(blank);

    clock_t start_time = clock();

    // --- Training loop ---
    float error = 1.0f;
    int   epoch = 0;
    for (; epoch < MAX_EPOCH && error > 0.00001f; epoch++) {
        float lr = BASE_LR / (1.0f + 0.001f * (float)epoch);
        error = 0.0f;

        shuffle(order, img_count);

        for (int idx = 0; idx < img_count; idx += (int)batch) {
            int cur = img_count - idx;
            if (cur > (int)batch) cur = (int)batch;

            for (int s = 0; s < cur; s++)
                memcpy(h_batch_in + (size_t)s * pixel_floats,
                    imgs_f[order[idx + s]], pixel_floats * sizeof(float));

            for (int s = 0; s < cur; s++)
                CUDA_CHECK(cudaMemcpy(
                    d_targets + (size_t)s * pixel_floats,
                    d_imgs[order[idx + s]],
                    pixel_floats * sizeof(float),
                    cudaMemcpyDeviceToDevice));

            forward_prop_batch(&net, h_batch_in, cur);
            float loss = backward_prop(&net, d_targets, lr);
            error += loss * ((float)cur / (float)img_count);
        }

        losses[epoch] = error;

        printf("\033[H");

        {
            int DISPLAY_IDX = rand() % img_count;
            for (unsigned int s = 0; s < net.batch; s++)
                memcpy(h_batch_in + (size_t)s * pixel_floats,
                    imgs_f[DISPLAY_IDX], pixel_floats * sizeof(float));

            forward_prop_batch(&net, h_batch_in, (int)net.batch);

            float* out_f = get_layer(&net, net.size - 1);
            unsigned char* out_img = float_to_img(out_f, pixel_floats);
            free(out_f);

            int plane = (int)(width * height);
            for (int i = 0; i < plane; i++) {
                frame_buf[i * 4 + 0] = imgs[DISPLAY_IDX][i * 3 + 0];
                frame_buf[i * 4 + 1] = imgs[DISPLAY_IDX][i * 3 + 1];
                frame_buf[i * 4 + 2] = imgs[DISPLAY_IDX][i * 3 + 2];
                frame_buf[i * 4 + 3] = 255;

                frame_buf[(i + plane) * 4 + 0] = out_img[i * 3 + 0];
                frame_buf[(i + plane) * 4 + 1] = out_img[i * 3 + 1];
                frame_buf[(i + plane) * 4 + 2] = out_img[i * 3 + 2];
                frame_buf[(i + plane) * 4 + 3] = 255;
            }

            if (draw_pictures) {
                UpdateTexture(screenTex, frame_buf);
                BeginDrawing();
                ClearBackground(BLACK);
                Vector2 pos = { 0 };
                DrawTextureEx(screenTex, pos, 0.0f,
                    (float)(WIN_W / (int)width), WHITE);
                DrawText(TextFormat("Epoch: %d", epoch + 1), 10, 10, 20, WHITE);
                DrawText(TextFormat("Loss: %.6f", error), 10, 40, 20, WHITE);
                EndDrawing();
            }

            if (output_images && epoch % 10 == 0) {
                unsigned char* bmp = (unsigned char*)malloc((size_t)pixel_floats * 2);
                if (bmp) {
                    memcpy(bmp, imgs[DISPLAY_IDX], pixel_floats);
                    memcpy(bmp + pixel_floats, out_img, pixel_floats);
                    char filename[32];
                    char filepath[APP_PATH_MAX];
                    snprintf(filename, sizeof(filename), "%04d.bmp", epoch);
                    if (path_join(filepath, sizeof(filepath), "image_output", filename))
                        write_bmp(filepath, (int)width, (int)height * 2, bmp);
                    free(bmp);
                }
            }

            free(out_img);
        }

        progress_bar(MAX_EPOCH, epoch + 1, 30);
        printf(" epoch %4d/%d  loss %.6f  lr %.6f\n", epoch + 1, MAX_EPOCH, error, lr);
        fflush(stdout);
    }

    clock_t end_time = clock();
    clock_t elapsed = end_time - start_time;

    printf("\nLoss curve:\n");
    for (int i = 0; i < epoch; i++) printf("%.6f, ", losses[i]);
    printf("\n");
    printf("Total training time: %.2f seconds\n", (float)elapsed / CLOCKS_PER_SEC);

    // -----------------------------------------------------------------------
    // Post-training: latent-space interpolation UI
    // -----------------------------------------------------------------------

    // --- Build decoder (shared weights from full autoencoder) ---
    int latent_size = (int)net.structure[net.size / 2];

    neural_net decoder;
    init_decoder(&decoder, &net);
    copy_decoder_weights(&net, &decoder);

    // Reuse h_batch_in for encoding single images (needs pixel_floats floats)
    // It was already allocated at batch*pixel_floats, so it's large enough.

    // --- Encode every training image once to get its latent vector ---
    float** latents = (float**)malloc(sizeof(float*) * (size_t)img_count);
    if (!latents) { fprintf(stderr, "OOM latents\n"); return 1; }
    for (int i = 0; i < img_count; i++) {
        memcpy(h_batch_in, imgs_f[i], pixel_floats * sizeof(float));
        forward_prop_batch(&net, h_batch_in, 1);
        latents[i] = get_layer(&net, net.size / 2);   // malloc'd by get_layer
    }

    // --- Decoder input buffer (one latent vector) ---
    float* latent_buf = (float*)malloc((size_t)latent_size * sizeof(float));
    if (!latent_buf) { fprintf(stderr, "OOM latent_buf\n"); return 1; }

    // --- Thumbnails: render each training image as a small texture ---
#define THUMB_SIZE   64
#define THUMB_COLS    8
// Window layout (fixed height, thumbnail panel scrolls):
//   [0 .. DECODED_H)                     decoded interpolated image
//   [DECODED_H .. DECODED_H+THUMB_VIS_H) thumbnail panel (scrollable)
//   [DECODED_H+THUMB_VIS_H .. bottom)    interpolation slider
#define DECODED_H     400
#define THUMB_VIS_H   192   // always exactly 3 rows visible
#define SLIDER_H       70

    int thumb_rows = (img_count + THUMB_COLS - 1) / THUMB_COLS;
    int thumb_total_h = thumb_rows * THUMB_SIZE;   // full virtual height
    int new_win_w = (THUMB_COLS * THUMB_SIZE > WIN_W) ? THUMB_COLS * THUMB_SIZE : WIN_W;
    int new_win_h = DECODED_H + THUMB_VIS_H + SLIDER_H;

    UnloadTexture(screenTex);
    free(frame_buf);
    SetWindowSize(new_win_w, new_win_h);

    // Decoded image texture
    Image dec_blank = GenImageColor(new_win_w, DECODED_H, BLACK);
    Texture2D decodedTex = LoadTextureFromImage(dec_blank);
    UnloadImage(dec_blank);
    frame_buf = (unsigned char*)malloc((size_t)new_win_w * DECODED_H * 4);
    memset(frame_buf, 0, (size_t)new_win_w * DECODED_H * 4);

    // One small texture per thumbnail
    Texture2D* thumbTex = (Texture2D*)malloc(sizeof(Texture2D) * (size_t)img_count);
    if (!thumbTex) { fprintf(stderr, "OOM thumbTex\n"); return 1; }
    {
        unsigned char* thumb_rgba = (unsigned char*)malloc((size_t)THUMB_SIZE * THUMB_SIZE * 4);
        for (int ii = 0; ii < img_count; ii++) {
            for (int py = 0; py < THUMB_SIZE; py++) {
                for (int px = 0; px < THUMB_SIZE; px++) {
                    int sx = px * (int)width / THUMB_SIZE;
                    int sy = py * (int)height / THUMB_SIZE;
                    const unsigned char* src_px = imgs[ii] + (sy * (int)width + sx) * 3;
                    unsigned char* dst_px = thumb_rgba + (py * THUMB_SIZE + px) * 4;
                    dst_px[0] = src_px[0];
                    dst_px[1] = src_px[1];
                    dst_px[2] = src_px[2];
                    dst_px[3] = 255;
                }
            }
            Image tmp = { thumb_rgba, THUMB_SIZE, THUMB_SIZE, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
            thumbTex[ii] = LoadTextureFromImage(tmp);
        }
        free(thumb_rgba);
    }

    // --- State ---
    int   sel_a = 0, sel_b = (img_count > 1) ? 1 : 0;
    float interp_t = 0.5f;
    bool  slider_drag = false;
    float thumb_scroll = 0.0f;   // pixels scrolled in thumbnail panel

    // Slider geometry — fixed at bottom band, computed once
    float sl_x = 20.0f;
    float sl_w = (float)(new_win_w - 40);
    float sl_h = 10.0f;
    // vertical centre of the track, 38px below the top of the slider band
    float sl_y = (float)(DECODED_H + THUMB_VIS_H + 38);

    SetTargetFPS(60);
    SetWindowTitle("CompressNet - Latent Interpolation");

    // Macro: forward-pass decoder with latent_buf, blit result into frame_buf/decodedTex
#define DECODE_TO_FRAMEBUF() \
    do { \
        forward_prop_batch(&decoder, latent_buf, 1); \
        float* _out = get_layer(&decoder, decoder.size - 1); \
        for (int _i = 0; _i < new_win_w * DECODED_H; _i++) { \
            int _dx = (_i % new_win_w) * (int)width  / new_win_w; \
            int _dy = (_i / new_win_w) * (int)height / DECODED_H; \
            int _oi = (_dy * (int)width + _dx) * 3; \
            float _r = _out[_oi+0]; float _g = _out[_oi+1]; float _b = _out[_oi+2]; \
            if (_r<0.f)_r=0.f; if (_r>1.f)_r=1.f; \
            if (_g<0.f)_g=0.f; if (_g>1.f)_g=1.f; \
            if (_b<0.f)_b=0.f; if (_b>1.f)_b=1.f; \
            frame_buf[_i*4+0]=(unsigned char)(_r*255.f+.5f); \
            frame_buf[_i*4+1]=(unsigned char)(_g*255.f+.5f); \
            frame_buf[_i*4+2]=(unsigned char)(_b*255.f+.5f); \
            frame_buf[_i*4+3]=255; \
        } \
        free(_out); \
        UpdateTexture(decodedTex, frame_buf); \
    } while(0)

    // Initial decode at t=0.5
    for (int k = 0; k < latent_size; k++)
        latent_buf[k] = latents[sel_a][k] * 0.5f + latents[sel_b][k] * 0.5f;
    DECODE_TO_FRAMEBUF();

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        bool mousePressed = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
        bool mouseDown = IsMouseButtonDown(MOUSE_LEFT_BUTTON);
        bool mouseReleased = IsMouseButtonReleased(MOUSE_LEFT_BUTTON);
        bool need_decode = false;

        // --- Scroll thumbnail panel with mouse wheel when hovering over it ---
        if (mouse.y >= DECODED_H && mouse.y < DECODED_H + THUMB_VIS_H) {
            thumb_scroll -= GetMouseWheelMove() * (float)THUMB_SIZE;
            float max_scroll = (float)(thumb_total_h - THUMB_VIS_H);
            if (max_scroll < 0.f) max_scroll = 0.f;
            if (thumb_scroll < 0.f)         thumb_scroll = 0.f;
            if (thumb_scroll > max_scroll)  thumb_scroll = max_scroll;
        }

        // --- Thumbnail picking (only within the visible panel) ---
        Rectangle panel_rect = { 0, (float)DECODED_H, (float)new_win_w, (float)THUMB_VIS_H };
        if (CheckCollisionPointRec(mouse, panel_rect)) {
            for (int ii = 0; ii < img_count; ii++) {
                int col = ii % THUMB_COLS;
                int row = ii / THUMB_COLS;
                float ty = (float)(DECODED_H + row * THUMB_SIZE) - thumb_scroll;
                // skip thumbnails scrolled outside the visible panel
                if (ty + THUMB_SIZE <= DECODED_H || ty >= DECODED_H + THUMB_VIS_H) continue;
                Rectangle tr = { (float)(col * THUMB_SIZE), ty, (float)THUMB_SIZE, (float)THUMB_SIZE };
                if (mousePressed && CheckCollisionPointRec(mouse, tr)) {
                    sel_a = ii; need_decode = true;
                }
                if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON) && CheckCollisionPointRec(mouse, tr)) {
                    sel_b = ii; need_decode = true;
                }
            }
        }

        // --- Interpolation slider ---
        // Wide hit-area (full height of slider band) for easy grabbing
        Rectangle sl_hit = { sl_x, (float)(DECODED_H + THUMB_VIS_H), sl_w, (float)SLIDER_H };
        if (mouseDown && CheckCollisionPointRec(mouse, sl_hit))
            slider_drag = true;
        if (mouseReleased)
            slider_drag = false;
        if (slider_drag) {
            float t = (mouse.x - sl_x) / sl_w;
            if (t < 0.f) t = 0.f;
            if (t > 1.f) t = 1.f;
            if (t != interp_t) { interp_t = t; need_decode = true; }
        }

        // --- Recompute latent & decode ---
        if (need_decode) {
            for (int k = 0; k < latent_size; k++)
                latent_buf[k] = latents[sel_a][k] * (1.f - interp_t)
                + latents[sel_b][k] * interp_t;
            DECODE_TO_FRAMEBUF();
        }

        // --- Draw ---
        BeginDrawing();
        ClearBackground(BLACK);

        // Decoded image (top DECODED_H pixels)
        {
            Rectangle src = { 0, 0, (float)new_win_w, (float)DECODED_H };
            Rectangle dst = { 0, 0, (float)new_win_w, (float)DECODED_H };
            Vector2   org = { 0, 0 };
            DrawTexturePro(decodedTex, src, dst, org, 0.f, WHITE);
            DrawText(TextFormat("A: img %d    B: img %d    t = %.2f", sel_a, sel_b, interp_t),
                10, 10, 18, YELLOW);
            DrawText("Left-click = A   Right-click = B   Scroll = browse", 10, 32, 14, LIGHTGRAY);
        }

        // Thumbnail panel background
		Color backg = { 20, 20, 20, 255 };
        DrawRectangle(0, DECODED_H, new_win_w, THUMB_VIS_H, backg);

        // Enable scissor so thumbnails don't bleed into decoded image or slider
        BeginScissorMode(0, DECODED_H, new_win_w, THUMB_VIS_H);
        for (int ii = 0; ii < img_count; ii++) {
            int col = ii % THUMB_COLS;
            int row = ii / THUMB_COLS;
            int tx = col * THUMB_SIZE;
            int ty = DECODED_H + row * THUMB_SIZE - (int)thumb_scroll;

            DrawTexture(thumbTex[ii], tx, ty, WHITE);
            if (ii == sel_a)
                DrawRectangleLines(tx, ty, THUMB_SIZE, THUMB_SIZE, GREEN);
            if (ii == sel_b)
                DrawRectangleLines(tx + 2, ty + 2, THUMB_SIZE - 4, THUMB_SIZE - 4, RED);
        }
        EndScissorMode();

        Color color;
        // Scrollbar indicator on the right edge of the thumbnail panel
        if (thumb_total_h > THUMB_VIS_H) {
            float bar_h = (float)THUMB_VIS_H * (float)THUMB_VIS_H / (float)thumb_total_h;
            float bar_y = (float)DECODED_H + thumb_scroll * (float)THUMB_VIS_H / (float)thumb_total_h;
            color = { 50, 50, 50, 255 };
            DrawRectangle(new_win_w - 6, DECODED_H, 6, THUMB_VIS_H, color);
			color = { 160 , 160, 160, 255 };
            DrawRectangle(new_win_w - 6, (int)bar_y, 6, (int)bar_h, color);
        }

        // Slider band
		color = { 30, 30, 30, 255 };
        DrawRectangle(0, DECODED_H + THUMB_VIS_H, new_win_w, SLIDER_H, color);
        DrawText("A                                                                                       B",
            (int)sl_x, DECODED_H + THUMB_VIS_H + 8, 14, LIGHTGRAY);
        // Track
        DrawRectangle((int)sl_x, (int)(sl_y - sl_h * 0.5f), (int)sl_w, (int)sl_h, DARKGRAY);
        // Fill A→knob in green
        
		color = { 80, 200, 80, 255 };
        DrawRectangle((int)sl_x, (int)(sl_y - sl_h * 0.5f),
            (int)(sl_w * interp_t), (int)sl_h, color);
        // Knob
        int knob_x = (int)(sl_x + sl_w * interp_t);
        DrawCircle(knob_x, (int)sl_y, 12, WHITE);
        DrawText(TextFormat("%.2f", interp_t), knob_x + 16, (int)(sl_y - 9), 16, YELLOW);

        EndDrawing();
    }

    // --- Cleanup ---
    for (int i = 0; i < img_count; i++) {
        free(latents[i]);
        UnloadTexture(thumbTex[i]);
    }
    free(latents);
    free(thumbTex);
    free(latent_buf);

    UnloadTexture(decodedTex);
    free(frame_buf);
    CloseWindow();

    for (int i = 0; i < img_count; i++) {
        free(imgs[i]);
        free(imgs_f[i]);
        cudaFree(d_imgs[i]);
    }
    free(imgs);
    free(imgs_f);
    free(d_imgs);
    free(order);
    free(losses);
    free(h_batch_in);
    cudaFree(d_targets);
    neural_net_free(&net);
    neural_net_free(&decoder);
    gpu_destroy();

    return 0;
}
