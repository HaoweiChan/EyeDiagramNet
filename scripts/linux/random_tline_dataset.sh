#!/bin/tcsh

# Generate a random transmission line dataset
# Usage: ./random_tline_dataset.sh

set python_cmd = ( python3 -m tests.data_generation.main generate --output "tests/data_generation/traces/transmission_lines_48lines.s96p" )

# number of random s-parameter files to generate
set k = 100
# number of lines
set n = 48

# generate s-parameters in parallel using a simple loop
foreach i (`seq 0 99`)
    ( ${python_cmd} --seed ${i} --n_lines ${n} )&
end

# wait for all background jobs to finish
wait

echo "Dataset generation complete."