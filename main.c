#include <stdio.h>
#include <mpi.h>


// The current function
double f(double x);

// Trapezoid rule
double Trap(
        double left_endpoint   /* in */,
        double right_endpoint  /* in */, 
        int trap_count         /* in */,
        double base_len        /* in */ ); 

// Get input for trapezoid program
void Get_input(
        int my_rank,    /* in  */ 
        int comm_sz,    /* in  */
        double * a_p,   /* out */
        double * b_p,   /* out */
        int * n_p       /* out */);

int main() {
    int my_rank, comm_sz, n = 1024, local_n;
    double a, b, h, local_a, local_b;
    double local_int, total_int;
    int source;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    Get_input(my_rank, comm_sz, &a, &b, &n);

    h = (b - a) / n;
    local_n = n / comm_sz; // Assume n evenly divides comm_sz

    local_a = a + my_rank * local_n * h; 
    local_b = local_a + local_n * h;
    local_int = Trap(local_a, local_b, local_n, h);

    MPI_Reduce(
        &local_int,     /* in  */
        &total_int,     /* out */ 
        1,              /* in  */
        MPI_DOUBLE,     /* in  */
        MPI_SUM,        /* in  */
        0,              /* in  */  // Destination process 
        MPI_COMM_WORLD  /* in  */
    );

    if (my_rank == 0) {
        printf("WIth n = %d trapezoids, our estimate\n", n);
        printf("of the integral from %f to %f = %.2lf\n", a, b, total_int);
    }

    MPI_Finalize();
    return 0;
}

double Trap(
        double left_endpoint   /* in */,
        double right_endpoint  /* in */, 
        int trap_count         /* in */,
        double base_len        /* in */ ) {
    double estimate, x;
    int i; 

    // This is basically the serial version of the trapezoid rule
    estimate = (f(left_endpoint) + f(right_endpoint)) / 2.0;
    for (i = 1; i <= trap_count -1; i++) {
        x = left_endpoint + i * base_len;
        estimate += f(x);
    }

    estimate = estimate * base_len;
    return estimate;
}

double f(double x) {
    return x * x + 2 * x + 3;
}


void Get_input(
    int my_rank,    /* in  */ 
    int comm_sz,    /* in  */
    double * a_p,   /* out */
    double * b_p,   /* out */
    int * n_p       /* out */
) {
    if (my_rank == 0) {
        printf("Enter a, b, and n\n");
        scanf("%lf %lf %d", a_p, b_p, n_p);
    }

    // Since souce process = 0 here, the MPI_Bcast understands that
    // if the process is != 0, then we will get the a_p, b_p and n_p values
    // from the void pointer, while if the rank == 0, then we will send the data
    // through the pointer. 
    //
    // This means that void * data_p (e.g., the a_p pointer) is an input argument
    // for the process equal to the destionation argument (i.e., process zero in this case)
    // and an output argument for all other processes
    //
    MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
}