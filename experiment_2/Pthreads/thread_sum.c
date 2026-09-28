#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define ARRAY_SIZE 100
#define NUM_THREADS 4

int array[ARRAY_SIZE];
long partial_sums[NUM_THREADS];

typedef struct {
    int thread_id;
    int start_index;
    int end_index;
} ThreadData;

void* compute_sum(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    long sum = 0;

    for (int i = data->start_index; i < data->end_index; i++) {
        sum += array[i];
    }

    partial_sums[data->thread_id] = sum;
    printf("Thread %d: calculated sum from index %d to %d = %ld\n", 
           data->thread_id, data->start_index, data->end_index - 1, sum);
           
    pthread_exit(NULL);
}

int main() {
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = i + 1;
    }

    int chunk_size = ARRAY_SIZE / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_index = i * chunk_size;
        thread_data[i].end_index = (i == NUM_THREADS - 1) ? ARRAY_SIZE : (i + 1) * chunk_size;

        pthread_create(&threads[i], NULL, compute_sum, &thread_data[i]);
    }

    long total_sum = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_sum += partial_sums[i];
    }

    printf("Final Total Sum = %ld (Expected: 5050)\n", total_sum);
    return 0;
}
