#! /usr/bin/env bash

function build_project(){
    mkdir -p build && cd build
    cmake ..
    make -j4
    cd ../
}

function run_project(){
    cd bin/
    ./main
    cd ../
}


function clean_project(){
    rm -rf log/
    rm -rf build/*
    rm -rf bin/*
}




function main() {
    local cmd="$1"
    shift
    case "${cmd}" in
        build)
            build_project
            ;;
        run)
            run_project
            ;;
        build_and_run)
            build_project && run_project
            ;;
        clean)
            clean_project
            echo "clean_project finished."
            ;;
        *)
            echo "${cmd} Unrecognized"
            ;;

    esac
}

main "$@"
