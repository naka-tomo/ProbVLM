import yaml

lines = open("requirements.txt").readlines()

pkgs_conda = []
pkgs_pip = []

for line in lines:
    line = line.strip()
    if line[0]!="#":
        if not "pypi_0" in line: 
            pkgs_conda.append( line )
        elif "pycocotools" in line:
            pkgs_conda.append( line.replace("=pypi_0", "") )
        elif "opencv-python" in line:
            pass
        else:
            pkgs_pip.append( line.replace("=pypi_0", "").replace("=", "==") )

output_yaml = {}
output_yaml["channels"] = ["defaults","conda-forge", "pytorch", "nvidia"]
output_yaml["dependencies"] = pkgs_conda + ["pip"] + [{"pip":pkgs_pip}]

with open("requirements.yaml", "w") as f:
    f.write(yaml.dump(output_yaml))