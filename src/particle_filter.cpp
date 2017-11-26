/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "map.h"
#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 100;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// Normal (Gaussian) distributions for x, y and theta
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	for(int i = 0; i < num_particles; ++i) {
		weights.push_back(1.0);
		Particle particle;
		particle.id = i;
		particle.x = x + dist_x(gen);
		particle.y = y + dist_y(gen);
		particle.theta = theta + dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	cout << "Predicion Start.........." << endl;

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	double vel_factor = velocity / yaw_rate;

	for(int i = 0; i < num_particles; ++i) {
		Particle particle = particles[i];

		//Case with yaw_rate > 0
		if(fabs(yaw_rate) > 0) {
			double angular_factor = particle.theta + yaw_rate * delta_t;
			particles[i].x += vel_factor * (sin(angular_factor) - sin(particle.theta));
			particles[i].y += vel_factor * (cos(particle.theta) - cos(angular_factor));
			particles[i].theta += yaw_rate * delta_t;
		}
		//Case with yaw_rate = 0
		else {
			particles[i].x += velocity * delta_t * cos(particle.theta);
			particles[i].y += velocity * delta_t * sin(particle.theta);
		}

		//Noise
	    particles[i].x += dist_x(gen);
	    particles[i].y += dist_y(gen);
	    particles[i].theta += dist_theta(gen);	

		//cout << particle.x << ", " << particle.y << ", " << particle.theta << endl;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	int closest_map_landmark_id;
	double min_distance, distance, delta_x, delta_y;

	//Transformed observations of current particle
	for(int oi = 0; oi < observations.size(); ++oi) {		
		LandmarkObs transf_observation = observations[oi];
		//Map landmarks within sensor range
		min_distance = 1000000.0;
		for(int pi = 0; pi < predicted.size(); ++pi) {
			LandmarkObs map_landmark = predicted[pi];

			distance = dist(transf_observation.x, transf_observation.y, 
							map_landmark.x, map_landmark.y);

			if(min_distance > distance) {
				min_distance = distance;
				closest_map_landmark_id = map_landmark.id;
			}
		}

		observations[oi].id = closest_map_landmark_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	cout << "UpdateWeights Start.........." << endl;
	double x_m, y_m, x_p, y_p, 
			theta_p, cos_theta, sin_theta,
			x_o, y_o, distance,
			multi_term_1, multi_term_2,
			delta_x, delta_y, obs_weight;

	int id_o;

	std::vector<LandmarkObs> transformed_observations;
	std::vector<LandmarkObs> map_landmarks_within_range;

	//Associate transformed observations to landmarks
	//And calculate particles weights

	//Precalculations to improve performance
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];
	double alpha = 1 / (2 * M_PI * sigma_x * sigma_y);

	for(int pi = 0; pi < num_particles; ++pi) {
	    //Assignments, precalculations and clearing of lists
	    Particle particle = particles[pi];
	    x_p = particle.x;
	    y_p = particle.y;
	    theta_p = particle.theta;
	    cos_theta = cos(theta_p);
	    sin_theta = sin(theta_p);
	    particles[pi].associations.clear();
	    particles[pi].sense_x.clear();
	    particles[pi].sense_y.clear();
	    transformed_observations.clear();
	    map_landmarks_within_range.clear();
	    particles[pi].weight = 1.0;

		//Transformations of observations to Map coords
		for(int oi = 0; oi < observations.size(); ++oi) {
			LandmarkObs observation = observations[oi];
			x_o = observation.x;
			y_o = observation.y;

			x_m = x_p + cos_theta * x_o - sin_theta * y_o;
			y_m = y_p + sin_theta * x_o + cos_theta * y_o;

			particles[pi].sense_x.push_back(x_m);
			particles[pi].sense_y.push_back(y_m);
		}

		//Create a list of transformed observations for this particle
		for(int toi = 0; toi < particles[pi].sense_x.size(); ++toi) {
			LandmarkObs landmark;
			landmark.x = particles[pi].sense_x[toi];
			landmark.y = particles[pi].sense_y[toi];
			landmark.id = -1;
			transformed_observations.push_back(landmark);
		}

		//Create a subset of the map landmarks filtered by sensor range
		for(int mi = 0; mi < map_landmarks.landmark_list.size(); ++mi) {
			Map::single_landmark_s map_landmark = map_landmarks.landmark_list[mi];

			distance = dist(x_p, y_p, map_landmark.x_f, map_landmark.y_f);
			
			if(distance <= sensor_range) {
				LandmarkObs landmark;
				landmark.x = map_landmark.x_f;
				landmark.y = map_landmark.y_f;
				landmark.id = map_landmark.id_i;
				map_landmarks_within_range.push_back(landmark);
			}
		}		

		//Assign nearest map landmark to each transformed observation for the current particle
		dataAssociation(map_landmarks_within_range, transformed_observations);

		//Calculate weight for the current particle
		for(int toi = 0; toi < transformed_observations.size(); ++toi) {
			LandmarkObs transformed_observation = transformed_observations[toi];

			x_o = transformed_observation.x;
			y_o = transformed_observation.y;
			//Id of nearest map landmark calculated with dataAssociation(...) function
			id_o = transformed_observation.id;

			particles[pi].associations.push_back(id_o);

			int map_landmarks_aoi = -1;
			if(id_o >= 0) {
				for (int aoi = 0; aoi < map_landmarks.landmark_list.size(); aoi++)
					if (map_landmarks.landmark_list[aoi].id_i == id_o) {
						map_landmarks_aoi = aoi;
						break;
					}

				x_m = map_landmarks.landmark_list[map_landmarks_aoi].x_f;
				y_m = map_landmarks.landmark_list[map_landmarks_aoi].y_f;
				delta_x = x_o - x_m;
				delta_y = y_o - y_m;

				//Multivariate Gaussian
				multi_term_1 = delta_x * delta_x  / (2 * sigma_x * sigma_x);
				multi_term_2 = delta_y * delta_y  / (2 * sigma_y * sigma_y);
				obs_weight = alpha * exp(-( multi_term_1 + multi_term_2 ));

				particles[pi].weight *= obs_weight;
			}
		}
	}
}

void ParticleFilter::resample() {
	cout << "Resample Start.........." << endl;
	double beta = 0.0;
	vector<Particle> resamp_particles;
	//Store weights in list

	vector<double> weights;
	for (int pi = 0; pi < num_particles; pi++) {
		weights.push_back(particles[pi].weight);
	}

	//Random index
	uniform_int_distribution<int> resamp_dist(0, num_particles - 1);
	int ri = resamp_dist(gen);

	//Max weight
	double max_weight = *max_element(weights.begin(), weights.end());

	//Uniform dist
	uniform_real_distribution<double> beta_dist(0.0, max_weight);

	//Resample
	for (int pi = 0; pi < num_particles; pi++) {
		beta += beta_dist(gen) * 2.0;

		while (beta > weights[ri]) {
		  beta -= weights[ri];
		  ri = (ri + 1) % num_particles;
		}

		resamp_particles.push_back(particles[ri]);
	}

	cout << "Resampled particles size.........." << endl;
	cout << resamp_particles.size() << endl;

	particles = resamp_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
