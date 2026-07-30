#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <openssl/bn.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define UPDATE_KEY_SIZE 16u
#define TRAILER_SIZE  0x4024u
#define REGION_SIZE   0x2000u
#define IO_CHUNK      (4u * 1024u * 1024u)

static const unsigned char partclone_magic[] = "partclone-image";

static void usage(const char *name)
{
    fprintf(stderr,
            "usage: %s --key FILE --input FILE --output FILE "
            "[--expect-partclone]\n",
            name);
}

static int read_exact_at(int fd, void *buffer, size_t length, off_t offset)
{
    unsigned char *out = buffer;
    size_t done = 0;

    while (done < length) {
        ssize_t got = pread(fd, out + done, length - done, offset + (off_t)done);
        if (got < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (got == 0) {
            errno = EIO;
            return -1;
        }
        done += (size_t)got;
    }
    return 0;
}

static int write_exact(int fd, const void *buffer, size_t length)
{
    const unsigned char *in = buffer;
    size_t done = 0;

    while (done < length) {
        ssize_t put = write(fd, in + done, length - done);
        if (put < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (put == 0) {
            errno = EIO;
            return -1;
        }
        done += (size_t)put;
    }
    return 0;
}

static int32_t little_i32(const unsigned char *p)
{
    uint32_t value = (uint32_t)p[0]
                   | ((uint32_t)p[1] << 8)
                   | ((uint32_t)p[2] << 16)
                   | ((uint32_t)p[3] << 24);
    return (int32_t)value;
}

static int select_bytes(const unsigned char *source, size_t source_size,
                        size_t base, int divisor, size_t count,
                        unsigned char *destination)
{
    int32_t seed;
    int remainder;
    size_t stride;
    size_t i;

    if (base > source_size || source_size - base < 4)
        return -1;
    seed = little_i32(source + base);
    remainder = seed % divisor;
    if (remainder < 0)
        return -1;
    stride = (size_t)remainder + 4;

    for (i = 0; i < count; ++i) {
        size_t index = base + (size_t)remainder + 8 + i * stride;
        if (index >= source_size)
            return -1;
        destination[i] = source[index];
    }
    return 0;
}

/*
 * Reproduce mbedTLS RSA public-mode PKCS#1 v1.5 recovery while enforcing the
 * type-1 block strictly. No recovered value is printed.
 */
static int recover_type1(const unsigned char *candidate, size_t candidate_size,
                         const unsigned char *fields, size_t wanted,
                         unsigned char *recovered)
{
    char modulus_hex[0x201];
    char exponent_hex[7];
    BIGNUM *n = NULL, *e = NULL, *c = NULL, *m = NULL;
    BN_CTX *bn_context = NULL;
    unsigned char encoded[0x200];
    int modulus_bytes;
    size_t separator;
    size_t i;
    int result = -1;

    for (i = 0; i < 0x206; ++i) {
        unsigned char value = fields[i];
        if (!((value >= '0' && value <= '9')
                || (value >= 'a' && value <= 'f')
                || (value >= 'A' && value <= 'F')))
            goto done;
    }
    memcpy(modulus_hex, fields, 0x200);
    modulus_hex[0x200] = '\0';
    memcpy(exponent_hex, fields + 0x200, 6);
    exponent_hex[6] = '\0';

    if (BN_hex2bn(&n, modulus_hex) == 0 || BN_hex2bn(&e, exponent_hex) == 0)
        goto done;
    modulus_bytes = BN_num_bytes(n);
    if (modulus_bytes < 64 || modulus_bytes > (int)sizeof(encoded)
            || (size_t)modulus_bytes > candidate_size)
        goto done;

    c = BN_bin2bn(candidate, modulus_bytes, NULL);
    m = BN_new();
    bn_context = BN_CTX_new();
    if (c == NULL || m == NULL || bn_context == NULL)
        goto done;
    if (BN_cmp(c, n) >= 0 || BN_mod_exp(m, c, e, n, bn_context) != 1)
        goto done;
    if (BN_bn2binpad(m, encoded, modulus_bytes) != modulus_bytes)
        goto done;

    if (encoded[0] != 0x00 || encoded[1] != 0x01)
        goto done;
    separator = 2;
    while (separator < (size_t)modulus_bytes && encoded[separator] == 0xff)
        ++separator;
    if (separator < 10 || separator >= (size_t)modulus_bytes
            || encoded[separator] != 0x00)
        goto done;
    ++separator;
    if ((size_t)modulus_bytes - separator != wanted)
        goto done;
    memcpy(recovered, encoded + separator, wanted);
    result = 0;

done:
    OPENSSL_cleanse(encoded, sizeof(encoded));
    BN_clear_free(n);
    BN_clear_free(e);
    BN_clear_free(c);
    BN_clear_free(m);
    BN_CTX_free(bn_context);
    return result;
}

static int recover_expected_hash(const unsigned char trailer[TRAILER_SIZE],
                                 unsigned char expected[SHA256_DIGEST_LENGTH])
{
    unsigned char candidate[0x200];
    unsigned char fields[0x206];
    const unsigned char *regions = trailer + 0x24;
    int result = -1;

    if (select_bytes(regions, 2 * REGION_SIZE, 0, 12,
                     sizeof(candidate), candidate) != 0)
        goto done;
    if (select_bytes(regions, 2 * REGION_SIZE, REGION_SIZE, 12,
                     sizeof(fields), fields) != 0)
        goto done;
    if (recover_type1(candidate, sizeof(candidate), fields,
                      SHA256_DIGEST_LENGTH, expected) != 0)
        goto done;
    result = 0;

done:
    OPENSSL_cleanse(candidate, sizeof(candidate));
    OPENSSL_cleanse(fields, sizeof(fields));
    return result;
}

static void make_counter_blocks(unsigned char *buffer, size_t blocks,
                                uint64_t first_block)
{
    size_t i;

    for (i = 0; i < blocks; ++i) {
        uint64_t number = first_block + i;
        unsigned char *p = buffer + i * 16;
        uint32_t low = (uint32_t)number;

        p[0] = (unsigned char)low;
        p[1] = (unsigned char)(low >> 8);
        p[2] = (unsigned char)(low >> 16);
        p[3] = (unsigned char)(low >> 24);
        memset(p + 4, 0, 12);
    }
}

int main(int argc, char **argv)
{
    const char *key_path = NULL;
    const char *input_path = NULL;
    const char *output_path = NULL;
    int expect_partclone = 0;
    int key_fd = -1, input_fd = -1, output_fd = -1;
    int output_created = 0;
    unsigned char *trailer = NULL;
    unsigned char *ciphertext = NULL, *counters = NULL, *keystream = NULL;
    unsigned char key[16] = {0};
    unsigned char trailer_digest[SHA256_DIGEST_LENGTH];
    unsigned char expected_hash[SHA256_DIGEST_LENGTH];
    unsigned char payload_hash[SHA256_DIGEST_LENGTH];
    EVP_MD_CTX *hash_context = NULL;
    EVP_CIPHER_CTX *aes_context = NULL;
    struct stat input_stat;
    off_t payload_size, offset;
    int status = EXIT_FAILURE;
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--key") == 0 && i + 1 < argc)
            key_path = argv[++i];
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (strcmp(argv[i], "--expect-partclone") == 0)
            expect_partclone = 1;
        else {
            usage(argv[0]);
            goto done;
        }
    }
    if (key_path == NULL || input_path == NULL || output_path == NULL) {
        usage(argv[0]);
        goto done;
    }
    if (strcmp(input_path, output_path) == 0) {
        fprintf(stderr, "input and output paths must differ\n");
        goto done;
    }

    key_fd = open(key_path, O_RDONLY | O_CLOEXEC);
    if (key_fd < 0) {
        perror("open update key");
        goto done;
    }
    {
        struct stat key_stat;
        if (fstat(key_fd, &key_stat) != 0) {
            perror("stat update key");
            goto done;
        }
        if (!S_ISREG(key_stat.st_mode)
                || key_stat.st_size != UPDATE_KEY_SIZE) {
            fprintf(stderr,
                    "update key must be a regular file of exactly %u bytes\n",
                    UPDATE_KEY_SIZE);
            goto done;
        }
    }
    if (read_exact_at(key_fd, key, UPDATE_KEY_SIZE, 0) != 0) {
        perror("read update key");
        goto done;
    }
    close(key_fd);
    key_fd = -1;

    input_fd = open(input_path, O_RDONLY | O_CLOEXEC);
    if (input_fd < 0) {
        perror("open input");
        goto done;
    }
    if (fstat(input_fd, &input_stat) != 0) {
        perror("stat input");
        goto done;
    }
    if (!S_ISREG(input_stat.st_mode) || input_stat.st_size <= TRAILER_SIZE) {
        fprintf(stderr, "input is not a sufficiently large regular file\n");
        goto done;
    }
    payload_size = input_stat.st_size - TRAILER_SIZE;

    trailer = malloc(TRAILER_SIZE);
    if (trailer == NULL
            || read_exact_at(input_fd, trailer, TRAILER_SIZE, payload_size) != 0) {
        perror("read trailer");
        goto done;
    }
    if (memcmp(trailer, "TER", 3) != 0) {
        fprintf(stderr, "missing LG TER trailer marker\n");
        goto done;
    }
    if (EVP_Digest(trailer + 0x24, 2 * REGION_SIZE, trailer_digest, NULL,
                   EVP_sha256(), NULL) != 1) {
        fprintf(stderr, "trailer SHA-256 calculation failed\n");
        goto done;
    }
    if (CRYPTO_memcmp(trailer + 4, trailer_digest, SHA256_DIGEST_LENGTH) != 0) {
        fprintf(stderr, "trailer SHA-256 mismatch\n");
        goto done;
    }
    if (recover_expected_hash(trailer, expected_hash) != 0) {
        fprintf(stderr, "trailer RSA/type-1 digest recovery failed\n");
        goto done;
    }
    OPENSSL_cleanse(trailer, TRAILER_SIZE);
    free(trailer);
    trailer = NULL;

    output_fd = open(output_path,
                     O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                     0600);
    if (output_fd < 0) {
        perror("create output (refusing overwrite)");
        goto done;
    }
    output_created = 1;

    ciphertext = malloc(IO_CHUNK);
    counters = malloc(IO_CHUNK + 16);
    keystream = malloc(IO_CHUNK + 16);
    hash_context = EVP_MD_CTX_new();
    aes_context = EVP_CIPHER_CTX_new();
    if (ciphertext == NULL || counters == NULL || keystream == NULL
            || hash_context == NULL || aes_context == NULL) {
        fprintf(stderr, "memory/context allocation failed\n");
        goto done;
    }
    if (EVP_DigestInit_ex(hash_context, EVP_sha256(), NULL) != 1
            || EVP_EncryptInit_ex(aes_context, EVP_aes_128_ecb(), NULL,
                                  key, NULL) != 1
            || EVP_CIPHER_CTX_set_padding(aes_context, 0) != 1) {
        fprintf(stderr, "OpenSSL initialization failed\n");
        goto done;
    }

    for (offset = 0; offset < payload_size;) {
        off_t remaining = payload_size - offset;
        size_t wanted = remaining > (off_t)IO_CHUNK
                      ? IO_CHUNK : (size_t)remaining;
        size_t blocks = (wanted + 15) / 16;
        size_t counter_bytes = blocks * 16;
        int produced = 0;
        size_t j;

        if (read_exact_at(input_fd, ciphertext, wanted, offset) != 0) {
            perror("read encrypted payload");
            goto done;
        }
        if (EVP_DigestUpdate(hash_context, ciphertext, wanted) != 1) {
            fprintf(stderr, "payload SHA-256 update failed\n");
            goto done;
        }
        make_counter_blocks(counters, blocks, (uint64_t)offset / 16);
        if (EVP_EncryptUpdate(aes_context, keystream, &produced,
                              counters, (int)counter_bytes) != 1
                || produced != (int)counter_bytes) {
            fprintf(stderr, "AES keystream generation failed\n");
            goto done;
        }
        for (j = 0; j < wanted; ++j)
            ciphertext[j] ^= keystream[j];
        if (offset == 0 && expect_partclone
                && (wanted < sizeof(partclone_magic) - 1
                    || memcmp(ciphertext, partclone_magic,
                              sizeof(partclone_magic) - 1) != 0)) {
            fprintf(stderr, "decrypted payload lacks Partclone magic\n");
            goto done;
        }
        if (write_exact(output_fd, ciphertext, wanted) != 0) {
            perror("write output");
            goto done;
        }
        offset += (off_t)wanted;
    }
    if (EVP_DigestFinal_ex(hash_context, payload_hash, NULL) != 1) {
        fprintf(stderr, "payload SHA-256 finalization failed\n");
        goto done;
    }
    if (CRYPTO_memcmp(payload_hash, expected_hash, SHA256_DIGEST_LENGTH) != 0) {
        fprintf(stderr, "encrypted-payload SHA-256 mismatch\n");
        goto done;
    }
    if (fsync(output_fd) != 0) {
        perror("fsync output");
        goto done;
    }
    if (close(output_fd) != 0) {
        output_fd = -1;
        perror("close output");
        goto done;
    }
    output_fd = -1;
    output_created = 0;
    fprintf(stderr,
            "validated trailer and encrypted-payload SHA-256; wrote %" PRIdMAX
            " plaintext bytes\n",
            (intmax_t)payload_size);
    status = EXIT_SUCCESS;

done:
    OPENSSL_cleanse(key, sizeof(key));
    OPENSSL_cleanse(expected_hash, sizeof(expected_hash));
    OPENSSL_cleanse(payload_hash, sizeof(payload_hash));
    if (trailer != NULL) {
        OPENSSL_cleanse(trailer, TRAILER_SIZE);
        free(trailer);
    }
    if (ciphertext != NULL) {
        OPENSSL_cleanse(ciphertext, IO_CHUNK);
        free(ciphertext);
    }
    if (counters != NULL) {
        OPENSSL_cleanse(counters, IO_CHUNK + 16);
        free(counters);
    }
    if (keystream != NULL) {
        OPENSSL_cleanse(keystream, IO_CHUNK + 16);
        free(keystream);
    }
    EVP_MD_CTX_free(hash_context);
    EVP_CIPHER_CTX_free(aes_context);
    if (key_fd >= 0)
        close(key_fd);
    if (input_fd >= 0)
        close(input_fd);
    if (output_fd >= 0)
        close(output_fd);
    if (output_created && output_path != NULL)
        unlink(output_path);
    return status;
}
