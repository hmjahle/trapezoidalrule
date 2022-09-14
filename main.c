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
        0,              /* in  */ 
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
        for (int dest = 1; dest < comm_sz; dest++) {
            MPI_Send(a_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(b_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(n_p, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    } else { /* my_rank != 0 */
        MPI_Recv(a_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(n_p, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}