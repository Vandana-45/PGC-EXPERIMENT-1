#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

void* print_hello(void* arg) {
    int thread_id = *((int*)arg);
    free(arg);
    printf("Thread %d: Executing inside thread routine\n", thread_id);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        int* id = malloc(sizeof(int));
        *id = i + 1;
        
        if (pthread_create(&threads[i], NULL, print_hello, id) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All %d threads have finished execution.\n", NUM_THREADS);
    return 0;
}
