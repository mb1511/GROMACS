#!/bin/bash
scp -r ../md_params user@bluecrystalp3.acrc.bris.ac.uk:~/md
ssh -X user@bluecrystalp3.acrc.bris.ac.uk 'cd md/test_md; qsub gmx_qsub.sh'
