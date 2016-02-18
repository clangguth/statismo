/*
 * This file is part of the statismo library.
 *
 * Author: Christoph Langguth (christoph.langguth@unibas.ch)
 *
 * Copyright (c) 2011-2015 University of Basel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * Neither the name of the project's author nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef STATISMO_MULTIVARIATENORMALDISTRIBUTION_H
#define STATISMO_MULTIVARIATENORMALDISTRIBUTION_H

#include "CommonTypes.h"
#include "StatismoUtils.h"

namespace statismo {
    class MultiVariateNormalDistribution {

    private:
        VectorType mean;
        MatrixType covariance;
        MatrixType covInv;


    public:

        typedef std::vector<VectorType> DataType;

        MultiVariateNormalDistribution(VectorType mean, MatrixType covariance): mean(mean), covariance(covariance) {
            if (!Utils::PseudoInverse(covariance, covInv)) {
                //FIXME: throw some exception
            }
        }

        const VectorType& GetMean() const {
            return mean;
        }

        const MatrixType& GetCovariance() const {
            return covariance;
        }

        float MahalanobisDistance(const VectorType& data) const {
            VectorType x0 = data - mean;

            float d = sqrt(x0.dot((covInv * x0)));
            return d;
        }

        static MultiVariateNormalDistribution EstimateFromData(const DataType& data) {
            unsigned int numSamples = data.size();

            VectorType mean = data[0];
            for (unsigned int i= 1; i < numSamples; ++i) {
                mean += data[i];
            }
            mean /= numSamples;

            MatrixType covariance(mean.rows(), mean.rows());
            for (int r = covariance.rows() - 1; r >= 0; --r) {
                for (int c =covariance.cols() - 1; c >= 0; --c ) {
                    covariance(r,c) = 0;
                }
            }

            for (DataType::const_iterator item = data.begin(); item != data.end(); item++) {
                VectorType diff = (*item) - mean;
                MatrixType outer = diff * diff.transpose();
                covariance = covariance + outer;
            }

            if (numSamples > 1) {
                covariance /= (numSamples - 1);
            }
            return MultiVariateNormalDistribution(mean, covariance);
        }
    };
}
#endif //STATISMO_MULTIVARIATENORMALDISTRIBUTION_H
