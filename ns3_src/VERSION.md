# NS-3 Build Info

## NS-3
- Version: ns-3.36
- Source: https://gitlab.com/nsnam/ns-3-dev
- Commit/Tag: ns-3.36

## 5G-LENA
- Version: v2.2.y
- Source: https://gitlab.com/cttc-lena/nr
- Git hash: d2ca73e  (run `git -C ~/ns3/ns-3.36/contrib/nr log -1 --oneline` to verify)

## Build command
```
cd ~/ns3/ns-3.36
./ns3 configure --enable-examples --build-profile=optimized
./ns3 build scratch/baseline_mmwave scratch/drl_beam_sim
```

## Known build fixes applied
See Implementation_Guidelines.docx Section 5 for the 7 build
errors resolved during setup.
