#include <iostream>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h> 
#include <time.h>
#include <sys/utime.h>
struct Simulation {
	int *board;
	int *win;
	int *rows;
	int *columns;
	int *possible;
};

__device__
void checkWin(Simulation *sim)
{
	sim->win[0] = 0;
	for (int j = 0; j < (sim->columns[0]); j++)
	{
		if (sim->board[0*sim->columns[0] + j] == sim->board[1*sim->columns[0] + j] 
			&& sim->board[0*sim->columns[0] + j] == sim->board[2*sim->columns[0] + j] && sim->board[0*sim->columns[0] + j] != 0)
		{
			sim->win[0] = sim->board[0*sim->columns[0] + j];
		}
	}
	for (int j = 0; j < (sim->rows[0]); j++)
	{
		if (sim->board[j*sim->columns[0] + 0] == sim->board[j*sim->columns[0] + 1] 
			&& sim->board[j*sim->columns[0] + 0] == sim->board[j*sim->columns[0] + 2] && sim->board[0 * sim->columns[0] + j] != 0)
		{
			sim->win[0] = sim->board[j*sim->columns[0] + 0];
		}
	}
	if (sim->board[0] == sim->board[4] && sim->board[0] == sim->board[8] && sim->board[0] != 0)
	{
		sim->win[0] = sim->board[0];
	}
	if (sim->board[2] == sim->board[4] && sim->board[2] == sim->board[6] && sim->board[2] != 0)
	{
		sim->win[0] = sim->board[2];
	}
}


void printBoard(Simulation sim)
{
	printf("columns[0]: %i", sim.columns[0]);
	printf("\nrows[0]: %i\n", sim.rows[0]);
	for (int i = 0; i < sim.rows[0]; i++)
	{
		for (int j = 0; j < sim.columns[0]; j++)
		{
			printf("  %i  ", sim.board[i * sim.columns[0] + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void createSim(Simulation *sim)
{
	cudaMallocManaged((void **)&sim->rows, sizeof(int));
	cudaMallocManaged((void **)&sim->columns, sizeof(int));
	sim->rows[0] = 6;
	sim->columns[0] = 7;
	cudaMallocManaged((void **)&sim->board, sizeof(int) * sim->columns[0] * sim->rows[0]);
	for (int i = 0; i < sim->columns[0] * sim->rows[0]; i++)
	{
		sim->board[i] = 0;
	}
	cudaMallocManaged((void **)&sim->possible, sizeof(int) * 8);
	cudaMallocManaged((void **)&sim->win, sizeof(int));
	sim->win[0] = 0;
	
	
}


__device__
void possibleMoves(Simulation *sim)
{
	int moves = 0;
	for (int i = 0; i < sim->columns[0]; i++)
	{
		for (int j = 0; j < sim->rows[0]; j++)
		{
			if (sim->board[j*sim->columns[0] + i] == 0)
			{
				moves++;
				sim->possible[moves] = j*sim->columns[0] + i;
			}
		}
	}
	sim->possible[0] = moves;
}

void resetBoard(Simulation *sim)
{
	//printf("A");
	//printf("\nrows[0]: %i, columns[0]: %i", sim.rows[0], sim.columns[0]);
	for (int i = 0; i < sim->columns[0] * sim->rows[0]; i++)
	{
		//printf("B");
		sim->board[i] = 0;
	}
	//printf("C");
	sim->win[0] = 0;
	sim->possible[0] = 1;
	//printf("D");
}

__device__
int randomMove(Simulation *sim, int player, unsigned int seed)
{
	float*r;
	int threads = 1;
	curandGenerator_t gen;
	curandCreateGenerator(&gen,
		CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,
		1234ULL);
	curandGenerateUniform(gen, r, 1);
	int val = 0;
	if (sim->possible[0] > 0)
	{
		int rInt = r[0] * sim->possible[0];
		val = sim->possible[rInt + 1];
		sim->board[val] = player;

	}
	else
	{
		val = 0;
	}
	
	/*curandState_t states;
	//printf("\nSeed Setter: %i", threadIdx.x + blockIdx.x);
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
	//	threadIdx.x + blockIdx.x, /* the sequence number should be different for each core (unless you want all
	//							  cores to get the same sequence of numbers for some reason - use thread id! */
	//	0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	//	&states);/* [blockIdx.x * threads + threadIdx.x]);*/
	/*int val = 0;
	if (sim->possible[0] > 0)
	{
		int r = curand(&states) % sim->possible[0];
		val = sim->possible[r + 1];
		sim->board[val] = player;

	}
	else
	{
		val = 0;
	}
	*/
	//free(&states);
	return val;

	//printf("The board value: %i\n", val);
}

__global__
void handleComputer(Simulation **sim, int seed)
{
	//printf("A");
	int player = -1;
	//printf("B");
	possibleMoves(sim[0]);
	//printf("C");
	randomMove(sim[0], player, seed);
	//printf("D");
}

__device__
int resultOfMove(Simulation *sim, int seed)
{
	int move = -1;
	int player = 1;
	if (sim->possible[0] != 0)
	{
		move = randomMove(sim, player, seed);
		if (player == 1) { player = -1; }
		else { player = 1; }
		possibleMoves(sim);
		//printf("9");
		checkWin(sim);
	}
	while (sim->win[0] == 0 && sim->possible[0] != 0)
	{
		seed++;
		//printf("\n7 + INDEX: %i", blockIdx.x * blockDim.x + threadIdx.x);
		randomMove(sim, player, seed);
		//printf("\n8 + INDEX: %i", blockIdx.x * blockDim.x + threadIdx.x);
		if (player == 1) { player = -1; }
		else { player = 1; }
		//printf("\n9 + INDEX: %i", blockIdx.x * blockDim.x + threadIdx.x);
		possibleMoves(sim);
		//printf("\n10 + INDEX: %i", blockIdx.x * blockDim.x + threadIdx.x);
		checkWin(sim);
		//printf("\n11 + INDEX: %i", blockIdx.x * blockDim.x + threadIdx.x);
	}
	//printf("\nDONE");
	return move;
}
__global__
void computerMove(Simulation **sim, int runs, int blocks, int threads, unsigned int seed, int *move)
{

	//printf("2");
	//printf("\nsim[0] rows[0]: %i\n", sim[0]->rows[0]);
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	/*printf("\nStride: %i", stride);
	printf("\nGridDim: %i", gridDim.x);
	printf("\nBlockDim: %i", blockDim.x);
	printf("\nIndex: %i", index);*/
	//printf("3");
	for (int i = index; i < runs; i += stride)
	{


		//seed++;
		//printf("5");
		possibleMoves(sim[i]);
		//printf("\n6 + INDEX: %i", index);
		int location = resultOfMove(sim[i], seed);
		//printf("\nIndex: %i\nLOCATION: %i", index, location);
		move[location] += sim[i]->win[0];
	}
	//printf("A");
}

__global__
void loopStarter(Simulation **sim)
{
	possibleMoves(sim[0]);
	checkWin(sim[0]);
}

int main(void)
{
	const int runs = 4096;
	int blockSize = 256;
	int numBlocks = (runs + blockSize - 1) / blockSize;
	// Allocate Unified Memory -- accessible from CPU or GPU
	Simulation **sim;// = new Simulation*();
					 //createSim(sim);
	cudaMallocManaged(&sim, sizeof(Simulation)*runs);
	for (int i = 0; i < runs; i++)
	{

		sim[i] = new Simulation();
		cudaMallocManaged(&sim[i], sizeof(Simulation));
		createSim(sim[i]);

	}
	sim[0]->possible[0] = 1;
	int wins = 0;
	int ties = 0;
	int losses = 0;
	for (int i = 0; i < 1000; i++)
	{
		while (sim[0]->win[0] == 0 && sim[0]->possible[0] > 0)
		{
			int *board;
			cudaMallocManaged(&board, sizeof(int) * sim[0]->columns[0] * sim[0]->rows[0]);
			for (int i = 0; i < sim[0]->columns[0] * sim[0]->rows[0]; i++)
			{
				board[i] = sim[0]->board[i];
			}
			int *move;// = new int[sim->columns[0] * sim->rows[0]];
			cudaMallocManaged(&move, sizeof(int) * sim[0]->columns[0] * sim[0]->rows[0]);
			for (int i = 0; i < sim[0]->columns[0] * sim[0]->rows[0]; i++)
			{
				move[i] = 0;
			}
			srand(time(NULL));
			//printf("Start");

			computerMove << <numBlocks, blockSize >> > (sim, runs, numBlocks, blockSize, rand(), move);

			cudaDeviceSynchronize();



			int maxValue = -1;
			int max = -1;
			for (int i = 0; i < runs; i++)
			{
				if (move[i] > maxValue)
				{
					maxValue = move[i];
					max = i;
				}
			}
			board[max] = 1;

			sim[0]->board = board;
			//printBoard(*sim[0]);
			handleComputer << <1, 1 >> > (sim, rand());
			cudaDeviceSynchronize();
			//printBoard(*sim[0]);
			for (int i = 1; i < runs; i++)
			{
				sim[i]->board = sim[0]->board;
			}
			loopStarter << <1, 1 >> > (sim);
			cudaDeviceSynchronize();
			//printBoard(*sim[0]);
			cudaFree(&board);
			cudaFree(&move);
			//printf("\nPossible Moves: %i and Win?: %i\n", sim[0]->possible[0], sim[0]->win[0]);
		}
		if (sim[0]->win[0] == 1)
		{
			wins++;
		}
		else if (sim[0]->win[0] == 0)
		{
			ties++;
		}
		else
		{
			losses++;
		}

		for (int i = 0; i < runs; i++)
		{
			resetBoard(sim[i]);
		}
		printf("\nITERATION: %i, TOTAL WINS: %i  Ties: %i  Losses: %i", i, wins, ties, losses);
	}
	printf("\nWins By Player One: %i  Ties: %i  Losses: %i... of %i total games.", wins, ties, losses, 1000);

	// Run kernel on 1M elements on the CPU



	//cudaFree(&sim);

	return 0;
}