{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [pkgs.python313Packages.numpy pkgs.python313Packages.matplotlib pkgs.python313Packages.pandas pkgs.git];

  languages.python.enable = true;

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };
}
