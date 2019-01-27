/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 10;  // TODO: Set the number of particles
  particles.clear();

  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  

  for ( int i = 0 ; i < num_particles ; i++ ){
    struct Particle p = {
      .id = i,
      .x = dist_x(gen),
      .y = dist_y(gen),
      .theta = dist_theta(gen),
      .weight = 1.0f,
    };
    //std::cout << "init["<<i<<"]="<<p.x<<","<<p.y<<std::endl;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  for ( unsigned i = 0 ; i < particles.size() ; i ++ ){
    auto& p = particles[i];
    double new_x = 0.;
    double new_y = 0.;
    double new_theta = 0.;
    if ( yaw_rate != 0. ){ 
      new_x = p.x + (velocity/yaw_rate) * ( std::sin(p.theta+yaw_rate*delta_t)
					    - std::sin(p.theta) );
      new_y = p.y + (velocity/yaw_rate) * ( std::cos(p.theta) 
					    - std::cos(p.theta+yaw_rate*delta_t) );
      new_theta = p.theta + yaw_rate*delta_t;
    }else{
      new_x = p.x + velocity * delta_t * std::cos(p.theta);
      new_y = p.y + velocity * delta_t * std::sin(p.theta);
      new_theta = p.theta;
    }

    if ( std::isnan( new_x ) ){
      std::cout << "theta " << p.theta + yaw_rate*delta_t << std::endl;
    }
    

    normal_distribution<double> dist_x(new_x, std_pos[0]);
    normal_distribution<double> dist_y(new_y, std_pos[1]);
    normal_distribution<double> dist_theta(new_theta, std_pos[2]);
    
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    //std::cout << "pred["<<i<<"]="<<p.x<<","<<p.y<<", "<<p.theta<<std::endl;
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    for( unsigned j = 0 ; j < observations.size() ;  j++ ){
      double min_dist=std::numeric_limits<double>::max();
      unsigned min_dist_id=0;
      for( unsigned i = 0 ; i < predicted.size() ; i++ ){
	double d = dist( predicted[i].x,
			 predicted[i].y,
			 observations[j].x,
			 observations[j].y );
	if ( d < min_dist ){
	  min_dist = d;
	  min_dist_id = i;
	}
    }
    observations[j].id=min_dist_id;
    //std::cout << "dataAssociation observations["<<j<<"].id="<<min_dist_id<<std::endl;
  }
}

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
		   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
    + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
  

  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double total_weight=0.;
  vector<LandmarkObs> transformed;
  transformed.resize(observations.size());
  for( unsigned i = 0 ; i < particles.size() ; i++ ){
    auto& p = particles[i];
    //Transform Observations
    for ( unsigned j = 0 ; j < observations.size() ; j++ ){
      const LandmarkObs &obs=observations[j];
      LandmarkObs pred={0};
      pred.id = j;
      // transform to map x coordinate
      pred.x = p.x + (std::cos(p.theta) * obs.x) - (std::sin(p.theta) * obs.y);
      // transform to map y coordinate
      pred.y = p.y + (std::sin(p.theta) * obs.x) + (std::cos(p.theta) * obs.y);
      transformed[j] = (pred);
      if ( std::isnan(transformed[j].x) ){
	std::cout << "particles[" << i << "]=" << p.x << ", " << p.y << ", " << p.theta << std::endl;
	std::cout << "particles[" << i << "]=" << particles[i].x << ", " << particles[i].y << std::endl;
	std::cout << "observations[" << j << "]=" << obs.x << ", " << obs.y << std::endl;
      }
    }
    
    //Is landmark prediction?
    //vector<LandmarkObs> landmarks;
    landmarks.clear();
    landmarks.resize(50);//sparse... may be 50 is enough
    int lks=0;
    for( int k = 0 ; k < map_landmarks.landmark_list.size() ; k++ ){
      if ( dist( p.x, p.y, 
		 map_landmarks.landmark_list[k].x_f,
		 map_landmarks.landmark_list[k].y_f) < sensor_range+10/*margin*/ ){
	LandmarkObs lo = {.id=map_landmarks.landmark_list[k].id_i,
			  .x=map_landmarks.landmark_list[k].x_f,
			  .y=map_landmarks.landmark_list[k].y_f
	};
	landmarks[lks++] = ( lo );
      }
    }
    // assiciate transformed to 
    dataAssociation(landmarks, transformed );
    {
      std::vector<int> associations;
      std::vector<double> sense_x;
      std::vector<double> sense_y;
      associations.resize(transformed.size());
      sense_x.resize(transformed.size());
      sense_y.resize(transformed.size());
      for ( unsigned j = 0 ; j < transformed.size() ; j++ ){
	associations[j]=(landmarks[transformed[j].id].id);
	sense_x[j]=(transformed[j].x);
	sense_y[j]=(transformed[j].y);
      }
      SetAssociations( p, associations, sense_x, sense_y );
    }
    //calc multivariate gausian
    double weight=1.0;
    for( unsigned j = 0 ; j < transformed.size() ; j++ ){
      double wn = multiv_prob(
			      std_landmark[0], std_landmark[1],
			      transformed[j].x, transformed[j].y,
			      landmarks[transformed[j].id].x,
			      landmarks[transformed[j].id].y
			      );
      if ( std::isnan(wn) ){
	std::cout << transformed[j].x<<","<<transformed[j].y<<
	  "associated to ["<<transformed[j].id<<"] "<<
	  landmarks[transformed[j].id].x <<","<<
	  landmarks[transformed[j].id].y <<std::endl;
	exit(-1);
      }
      weight *= wn;
    }
    p.weight = weight;//update
    total_weight += weight;
  }
  weights.resize(particles.size());
  for( unsigned i = 0 ; i < particles.size() ; i++ ){
     auto& p = particles[i];
     p.weight = p.weight/total_weight;
     weights[i] = (p.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> resampled;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(),weights.end());
  resampled.resize(particles.size());
  for( int i = 0 ; i < particles.size() ; i++ ){
    int idx = d(gen);
    resampled[i] = ( particles[idx] );
  }
  // for( int i = 0 ; i < particles.size(); i++ ){
  //   std::cout << "pre:particles[" << i << "].weight=" << particles[i].weight << std::endl;
  // }
  particles = resampled;/* copy */
  // for( int i = 0 ; i < particles.size(); i++ ){
  //   std::cout << "particles[" << i << "].weight=" << particles[i].weight << std::endl;
  // }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
