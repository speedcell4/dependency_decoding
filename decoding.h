// Copyright (c) 2012-2015 Andre Martins
// All Rights Reserved.
//
// This file is part of TurboParser 2.3.
//
// TurboParser 2.3 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TurboParser 2.3 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with TurboParser 2.3.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <vector>

// Decoder for the basic model; it finds a maximum weighted arborescence
// using Edmonds' algorithm (which runs in O(n^2)).
void c_chu_liu_edmonds(std::vector<bool> *disabled,
                       std::vector<std::vector<int> > *candidate_heads,
                       std::vector<std::vector<double> > *candidate_scores,
                       std::vector<int> *heads,
                       double *value);

void batch_c_chu_liu_edmonds(
    std::vector<std::vector<bool> > *disabled,
    std::vector<std::vector<std::vector<int> > > *candidate_heads,
    std::vector<std::vector<std::vector<double> > > *candidate_scores,
    std::vector<std::vector<int> > *heads,
    std::vector<double> *values);