import numpy as np
from datetime import datetime
import os


class CF:
    def __init__(self):
        """ Collaborative Filtering
        K is the neighborhood number
        users and users_r are dictionaries for user_id and user_index
        movies and movies_r are dictionaries for movie_id and movie_index
        user_sim is the dictionary for user similarity
        user_sm is the matrix for user similarity
        movie_user_rating, dictionary for ratings
        movie_sums, sum of the ratings for each film
        user_sums, sum of the ratings given by each user
        movie_counts, number of users that give rating for each film
        user_counts, number of ratings given by a user
        user_avg, average rating given by a user
        movie_avg, average rating of a film

        """
        self.K = 20
        self.user_sim = {}
        self.usc = 0
        self.users = {}
        self.movies = {}
        self.users_r = {}
        self.movies_r = {}
        self.movie_user_rating = {}
        self.movie_users = {}
        self.cur_user_index = 0
        self.cur_movie_index = 0
        self.movie_sums = []
        self.user_sums = []
        self.user_counts = []
        self.movie_counts = []
        self.user_avg = []
        self.movie_avg = []

    def add_rating(self, movie_id, user_id, rating):
        """
        Adds ratings to the movie_user_rating dictionary
        Keeps the number and sums of ratings for movies and users
        movie_sums, user_sums, movie_counts, user_counts
        """
        if user_id in self.users:
            user_index = self.users[user_id]
        else:
            self.users[user_id] = self.cur_user_index
            self.users_r[self.cur_user_index] = user_id
            user_index = self.cur_user_index
            self.user_sums.append(0)
            self.user_counts.append(0)
            self.cur_user_index += 1
        if movie_id in self.movies:
            movie_index = self.movies[movie_id]
        else:
            self.movies[movie_id] = self.cur_movie_index
            self.movies_r[self.cur_movie_index] = movie_id
            movie_index = self.cur_movie_index
            self.movie_sums.append(0)
            self.movie_counts.append(0)
            self.cur_movie_index += 1
        if movie_index in self.movie_user_rating:
            user_rating = self.movie_user_rating[movie_index]
        else:
            user_rating = {}
            self.movie_user_rating[movie_index] = user_rating
        if user_index in user_rating:
            print('dublicate rating', user_rating[user_index], rating)
        else:
            user_rating[user_index] = rating
            self.user_sums[user_index] += rating
            self.movie_sums[movie_index] += rating
            self.user_counts[user_index] += 1
            self.movie_counts[movie_index] += 1

    def print_rating(self, max_num=5):
        """
        Prints a subset of the given ratings
        """
        for i in range(max_num):
            if i in self.movie_user_rating:
                user_rating = self.movie_user_rating[i]
                for j in range(max_num):
                    if j in user_rating:
                        print(i, j, user_rating[j])
                        print(self.movies_r[i], self.users_r[
                              j], user_rating[j])

    def convert_matrix(self):
        """
        Converts rating hashtable to matrix, in order to run faster
        Calculates user_avg and movie_avg
        """
        self.user_sums = np.array(self.user_sums)
        self.movie_sums = np.array(self.movie_sums)
        self.user_counts = np.array(self.user_counts)
        self.movie_counts = np.array(self.movie_counts)
        self.user_avg = self.user_sums / self.user_counts
        self.movie_avg = self.movie_sums / self.movie_counts
        M = np.zeros((len(self.movies), len(self.users)))
        for movie_index in self.movie_user_rating:
            user_rating = self.movie_user_rating[movie_index]
            for user_index in user_rating:
                rating = user_rating[user_index]
                M[movie_index, user_index] = rating
        self.M = M

    def validate(self):
        """
        Matches the sums from the matrix and the sums from the dictionary
        """
        self.movie_sums_m = self.M.sum(axis=1)
        self.movie_sums = np.array(self.movie_sums)
        print('movie', self.movie_sums_m, self.movie_sums,
              self.movie_sums_m == self.movie_sums)
        self.user_sums_m = self.M.sum(axis=0)
        self.user_sums = np.array(self.user_sums)
        print('user', self.user_sums_m, self.user_sums,
              self.user_sums_m == self.user_sums)

    def pearson(self, user_index, user_index2):
        """
        Calculates similarities of two users
        If no common movie, returns -1
        If a users all ratings are same, returns -1
        Otherwise returns the similarity
        """
        m = self.M[:, user_index] * self.M[:, user_index2]
        movies = np.nonzero(m)[0]
        if movies.shape[0] == 0:
            return -1
        movie_ratings = self.M[np.ix_(movies, [user_index, user_index2])]
        mr1 = movie_ratings[:, 0] - self.user_avg[user_index]
        mr2 = movie_ratings[:, 1] - self.user_avg[user_index2]
        if (np.all(mr1 == 0) or np.all(mr2 == 0)):
            return -1
        pearson_1 = (mr1 * mr2).sum()
        pearson_2 = np.sqrt((mr1 ** 2).sum() * (mr2 ** 2).sum())
        pearson_sim = pearson_1 / pearson_2
        return pearson_sim

    def users_rated_movie(self, movie_index):
        """
        Returns the indices of users that gave rating for the given movie
        """
        if movie_index in self.movie_users:
            return self.movie_users[movie_index]
        user_indices = np.nonzero(self.M[movie_index, :])[0]
        self.movie_users[movie_index] = user_indices
        return user_indices

    def predict(self, movie_id, user_id):
        """
        Finds the index of the movie that movie_id belongs
        Finds all users that gave rating to that movie
        Finds the index of the user that user_id belongs
        Calculates the similarities of the user, and the other users that
            gave rating to the movie
        Selects the K neigbor user with the highest similarity
        """
        movie_index = self.movies[movie_id]
        user_index = self.users[user_id]
        other_users = self.users_rated_movie(movie_index)
        sim_mat = self.user_sm[user_index, other_users]
        near_ind = np.argsort(sim_mat)[::-1][:self.K]
        near_pearson = sim_mat[near_ind]
        user_ind = other_users[near_ind]

        ru_avg = self.user_avg[user_ind]
        rui = self.M[movie_index, user_ind]

        Euk_r = np.sum((rui - ru_avg) * near_pearson)
        Euk_w = np.sum(near_pearson)
        prediction = self.user_avg[user_index] + (Euk_r / Euk_w)
        return prediction

    def train(self):
        """
        Read file
        Convert ratings to matrix
        Creates user similarity matrices
        """
        self.t0 = datetime.now()
        ftrain = open('TrainingRatings.txt')
        print('train:')
        i = 0
        print_int = 250000
        for line in ftrain:
            line_array = line.split(',')
            movie_id = line_array[0]
            user_id = line_array[1]
            rating = float(line_array[2])
            self.add_rating(movie_id, user_id, rating)
            i += 1
            if i > 0 and i % print_int == 0:
                print(i / print_int, 'time:', str(datetime.now() - self.t0))
        print('data read, ', str((datetime.now() - self.t0)))
        self.convert_matrix()
        print('converted to matrix, ', str((datetime.now() - self.t0)))
        self.build_user_sim()
        print('user sim matrix created, ', str((datetime.now() - self.t0)))

    def build_user_sim(self):
        """
        Creates user similarity matrices
        Calculates the similarities for each user and the subsequent users
        Result matrix is symmetric
        Similarity between a user and self is 1
        """
        userc = len(self.user_sums)
        self.user_sm = np.zeros((userc, userc))
        print('building user similarity matrix')
        print_int = 1
        for i in range(userc):
            self.user_sm[i, i] = 1
            for j in range(i + 1, userc):
                pear_sim = self.pearson(i, j)
                self.user_sm[i, j] = pear_sim
                self.user_sm[j, i] = pear_sim
            if i > 0 and i % print_int == 0:
                print(i, str((datetime.now() - self.t0)))
                if print_int < 1000:
                    print_int *= 2
                elif print_int > 1000:
                    print_int = 1000

    def test(self):
        """
        Read test file
        Creates output files
        """
        ftest = open('TestingRatings.txt')
        fresult = open('PredictRatings.txt', 'w')
        fcompare = open('CompareRatings.txt', 'w')
        frecom = open('RecommendMovie.txt', 'w')
        print('test:')
        print_int = 100
        i = 0
        SAE = 0.0
        for line in ftest:
            line_array = line.split(',')
            movie_id = line_array[0]
            user_id = line_array[1]
            rating = float(line_array[2])
            prediction = self.predict(movie_id, user_id)
            SAE += np.abs(prediction - rating)
            fresult.write('%s,%s,%f\n' % (movie_id, user_id, prediction))
            fcompare.write('%s,%s,%f,%f\n' %
                           (movie_id, user_id, prediction, rating))
            if prediction > 4.0:
                frecom.write('%s,%s\n' % (movie_id, user_id))
            i += 1
            if i > 0 and i % print_int == 0:
                print(i, SAE, (SAE / i)), 'time:',
                str((datetime.now() - self.t0))

        fresult.close()
        fcompare.close()
        frecom.close()
        print(i, SAE, (SAE / i), 'time:', str((datetime.now() - self.t0)))

    def run(self):
        self.train()
        self.test()


"""
After the user similarity matrix is created
"""


def predict_pos(cf, movie_id, user_id):
    """
    Finds the index of the movie that movie_id belongs
    Finds all users that gave rating to that movie
    Finds the index of the user that user_id belongs
    Calculates the similarities of the user, and the other users that
        gave rating to the movie
    Selects the K neigbor user with the highest similarity
    """
    movie_index = cf.movies[movie_id]
    user_index = cf.users[user_id]
    other_users_all = cf.users_rated_movie(movie_index)
    sim_mat_all = cf.user_sm[user_index, other_users_all]
    sim_ind = sim_mat_all > 0
    # If all similarities are negative
    if sim_ind.sum() == 0:
        other_users = other_users_all
        sim_mat = sim_mat_all
    else:
        other_users = other_users_all[sim_ind]
        sim_mat = sim_mat_all[sim_ind]
    near_ind = np.argsort(sim_mat)[::-1][:cf.K]
    near_pearson = sim_mat[near_ind]
    user_ind = other_users[near_ind]

    ru_avg = cf.user_avg[user_ind]
    rui = cf.M[movie_index, user_ind]

    Euk_r = np.sum((rui - ru_avg) * near_pearson)
    Euk_w = np.sum(near_pearson)
    prediction = cf.user_avg[user_index] + (Euk_r / Euk_w)
    return prediction


def test(cf, K=20):
    """
    Read test file
    Creates output files
    """
    t0 = datetime.now()
    if K != 20 and K != cf.K:
        cf.K = K
    outfolder = 'result_%s' % K
    if not os.path.exists("./%s" % outfolder):
        os.makedirs("./%s" % outfolder)
    ftest = open('TestingRatings.txt')
    fresult = open(outfolder + '/PredictRatings.txt', 'w')
    fcompare = open(outfolder + '/CompareRatings.txt', 'w')
    frecom = open(outfolder + '/RecommendMovie.txt', 'w')
    fresultline = open(outfolder + '/ResultLine.txt', 'w')
    fallresults = open('allresults.txt', 'a')
    fresultline.write('i,SAE,MAE,SSE,RMSE,0,1,2,3,4,5,duration\n')
    print_int = 1000
    i = 0
    SAE = 0.0
    SSE = 0.0
    erm = [0] * 6
    for line in ftest:
        line_array = line.split(',')
        movie_id = line_array[0]
        user_id = line_array[1]
        rating = float(line_array[2])
        prediction = predict_pos(cf, movie_id, user_id)
        error = np.abs(prediction - rating)
        SAE += error
        SSE += np.square(error)
        ier = int(np.round(error))
        if ier < 0 or ier > 4:
            print(ier, error, prediction, rating, movie_id, user_id)
            ier = 5
        erm[ier] += 1
        fresult.write('%s,%s,%f\n' % (movie_id, user_id, prediction))
        fcompare.write('%s,%s,%f,%f\n' %
                       (movie_id, user_id, prediction, rating))
        if prediction > 4.0:
            frecom.write('%s,%s\n' % (movie_id, user_id))
        i += 1
        if i > 0 and i % print_int == 0:
            RMSE = np.sqrt(SSE / i)
            fresultline.write('%d,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d,%s\n' % (
                i, SAE, (SAE / i), SSE, RMSE,
                erm[0], erm[1], erm[2], erm[3], erm[4], erm[5],
                str((datetime.now() - t0))))

    fresult.close()
    fcompare.close()
    frecom.close()
    RMSE = np.sqrt(SSE / i)
    print(K, SAE, (SAE / i),
          SSE, RMSE, erm, 'time:',
          str((datetime.now() - t0)))
    fresultline.write('END,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d,%s\n' % (
        SAE, (SAE / i), SSE, RMSE,
        erm[0], erm[1], erm[2], erm[3], erm[4], erm[5],
        str((datetime.now() - t0))))
    fallresults.write('%d,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d,%s\n' % (
        K, SAE, (SAE / i), SSE, RMSE,
        erm[0], erm[1], erm[2], erm[3], erm[4], erm[5],
        str((datetime.now() - t0))))
    fresultline.close()
    fallresults.close()


def testfork(cf, mink=10, maxk=50):
    """
    Calculate results for different K values
    """
    fallresults = open('allresults.txt', 'w')
    fallresults.write('K,SAE,MAE,SSE,RMSE,0,1,2,3,4,5,duration\n')
    fallresults.close()
    for K in range(mink, maxk + 1):
        test(cf, K)


def main():
    cf = CF()
    cf.train()
    cf.test()
    testfork(cf, mink=10, maxk=100)


if __name__ == '__main__':
    main()
