extern crate oxigen;
extern crate rand;
use std::fs::File;

use oxigen::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::thread_rng;
use std::fmt::Display;
#[derive(Clone, Debug)]
struct Sphere(Vec<f64>);

impl Display for Sphere {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self.0)
    }
}

impl Genotype<f64> for Sphere {
    type ProblemSize = usize;

    fn iter(&self) -> std::slice::Iter<f64> {
        self.0.iter()
    }
    fn into_iter(self) -> std::vec::IntoIter<f64> {
        self.0.into_iter()
    }
    fn from_iter<I: Iterator<Item = f64>>(&mut self, genes: I) {
        self.0 = genes.collect();
    }
    fn generate(size: &Self::ProblemSize) -> Self {
        let mut individual: Vec<f64> = Vec::with_capacity(*size as usize);
        let mut rgen = thread_rng();
        let dist = Uniform::from(-5.12..=5.12);
        for _i in 0..*size {
            individual.push(dist.sample(&mut rgen));
        }
        Sphere(individual)
    }
    fn fitness(&self) -> f64 {
        let r: f64 = self.0.iter().map(|x| x * x).sum();
        1.0 / r
    }
    fn mutate(&mut self, _rgen: &mut SmallRng, index: usize) {
        let dist = Uniform::from(-5.12..=5.12);
        let mut rgen = thread_rng();
        self.0[index] = dist.sample(&mut rgen);
    }
    fn is_solution(&self, fitness: f64) -> bool {
        fitness <= 0.01
    }
}

fn main() {
    let problem_size: usize = std::env::args()
        .nth(1)
        .expect("Enter a number bigger than 1")
        .parse()
        .expect("Enter a number bigger than 1");
    let population_size = problem_size * 8;
    let log2 = (f64::from(problem_size as u32) * 4_f64).log2().ceil();
    let progress_log = File::create("progress.csv").expect("Error creating progress log file");
    let population_log =
        File::create("population.txt").expect("Error creating population log file");
    let (solutions, generation, _progress, _population) = GeneticExecution::<f64, Sphere>::new()
        .progress_log(100, progress_log)
        .population_log(500, population_log)
        .population_size(population_size)
        .genotype_size(problem_size)
        .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
            start: f64::from(problem_size as u32) / (8_f64 + 2_f64 * log2) / 100_f64,
            bound: 0.005,
            coefficient: -0.0002,
        })))
        .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
            start: log2 - 2_f64,
            bound: log2 / 1.5,
            coefficient: -0.0005,
        })))
        .select_function(Box::new(SelectionFunctions::Cup))
        .stop_criterion(Box::new(StopCriteria::Generation(10000)))
        .run();

    println!("Finished in the generation {}", generation);
    for sol in &solutions {
        println!("{}", sol);
    }
}
