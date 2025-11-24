#!/usr/bin/env python3
"""
GPU4PySCF Test Suite
Tests GPU functionality, basic DFT calculations, and performance comparisons
"""

import sys
import time
import traceback
from pyscf import gto, dft
import numpy as np
from test_utils import cleanup_gpu_memory, full_cleanup

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_success(message):
    """Print success message"""
    print(f"✓ {message}")

def print_error(message):
    """Print error message"""
    print(f"✗ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ {message}")

def test_gpu_detection():
    """Test 1: GPU Detection and CUDA Setup"""
    print_header("Test 1: GPU Detection and CUDA Setup")

    try:
        import cupy as cp
        print_success("CuPy imported successfully")

        # Check CUDA availability
        cuda_available = cp.cuda.is_available()
        print_info(f"CUDA available: {cuda_available}")

        if not cuda_available:
            print_error("CUDA is not available!")
            return False

        # Get GPU information
        device = cp.cuda.Device()
        print_info(f"GPU Device ID: {device.id}")
        print_info(f"GPU Name: {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}")

        # Get memory information
        mem_info = cp.cuda.runtime.memGetInfo()
        free_mem_gb = mem_info[0] / 1024**3
        total_mem_gb = mem_info[1] / 1024**3
        print_info(f"GPU Memory: {free_mem_gb:.2f} GB free / {total_mem_gb:.2f} GB total")

        # Test basic CuPy operation
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        expected = cp.array([5, 7, 9])

        if cp.allclose(c, expected):
            print_success("Basic CuPy operations work correctly")
        else:
            print_error("CuPy operations failed")
            return False

        # Cleanup GPU memory after test
        cleanup_gpu_memory(verbose=False)
        return True

    except Exception as e:
        print_error(f"GPU detection failed: {str(e)}")
        traceback.print_exc()
        return False

def test_gpu4pyscf_import():
    """Test 2: GPU4PySCF Import"""
    print_header("Test 2: GPU4PySCF Import")

    try:
        from gpu4pyscf import dft as gpu_dft
        print_success("gpu4pyscf.dft imported successfully")

        # Check available methods
        print_info("Available GPU-accelerated methods:")
        methods = [attr for attr in dir(gpu_dft) if not attr.startswith('_')]
        for method in methods[:10]:  # Show first 10
            print(f"  - {method}")
        if len(methods) > 10:
            print(f"  ... and {len(methods) - 10} more")

        return True

    except Exception as e:
        print_error(f"GPU4PySCF import failed: {str(e)}")
        traceback.print_exc()
        return False

def test_basic_dft_cpu():
    """Test 3: Basic DFT Calculation on CPU"""
    print_header("Test 3: Basic DFT Calculation on CPU (Water Molecule)")

    try:
        # Create water molecule
        mol = gto.M(
            atom='''
            O  0.0000  0.0000  0.1173
            H  0.0000  0.7572 -0.4692
            H  0.0000 -0.7572 -0.4692
            ''',
            basis='def2-svp',
            verbose=0
        )

        print_info(f"Molecule: H2O")
        print_info(f"Basis: {mol.basis}")
        print_info(f"Number of electrons: {mol.nelectron}")
        print_info(f"Number of basis functions: {mol.nao}")

        # Run DFT calculation
        print_info("Running DFT calculation on CPU...")
        start_time = time.time()

        mf_cpu = dft.RKS(mol)
        mf_cpu.xc = 'B3LYP'
        mf_cpu.verbose = 0
        energy_cpu = mf_cpu.kernel()

        cpu_time = time.time() - start_time

        print_success(f"CPU calculation completed in {cpu_time:.3f} seconds")
        print_info(f"Total energy: {energy_cpu:.8f} Hartree")

        # Cleanup CPU calculation objects
        full_cleanup(mf_cpu, verbose=False)

        return True, energy_cpu, cpu_time, mol

    except Exception as e:
        print_error(f"CPU DFT calculation failed: {str(e)}")
        traceback.print_exc()
        return False, None, None, None

def test_basic_dft_gpu(mol, energy_cpu):
    """Test 4: Basic DFT Calculation on GPU"""
    print_header("Test 4: Basic DFT Calculation on GPU (Water Molecule)")

    try:
        from gpu4pyscf import dft as gpu_dft

        print_info("Running DFT calculation on GPU...")
        start_time = time.time()

        mf_gpu = gpu_dft.RKS(mol)
        mf_gpu.xc = 'B3LYP'
        mf_gpu.verbose = 0
        energy_gpu = mf_gpu.kernel()

        gpu_time = time.time() - start_time

        print_success(f"GPU calculation completed in {gpu_time:.3f} seconds")
        print_info(f"Total energy: {energy_gpu:.8f} Hartree")

        # Compare energies
        energy_diff = abs(energy_gpu - energy_cpu)
        print_info(f"Energy difference (GPU - CPU): {energy_diff:.2e} Hartree")

        if energy_diff < 1e-6:
            print_success("GPU and CPU energies match within tolerance")
        else:
            print_error(f"Energy difference too large: {energy_diff}")
            full_cleanup(mf_gpu, verbose=False)
            return False, None

        # Cleanup GPU calculation objects
        full_cleanup(mf_gpu, verbose=False)

        return True, gpu_time

    except Exception as e:
        print_error(f"GPU DFT calculation failed: {str(e)}")
        traceback.print_exc()
        return False, None

def test_performance_comparison(cpu_time, gpu_time):
    """Test 5: Performance Comparison"""
    print_header("Test 5: Performance Comparison")

    if cpu_time is None or gpu_time is None:
        print_error("Cannot compare performance: missing timing data")
        return

    speedup = cpu_time / gpu_time

    print_info(f"CPU time: {cpu_time:.3f} seconds")
    print_info(f"GPU time: {gpu_time:.3f} seconds")
    print_info(f"Speedup: {speedup:.2f}x")

    if speedup > 1.0:
        print_success(f"GPU is {speedup:.2f}x faster than CPU")
    elif speedup > 0.5:
        print_info("GPU and CPU performance are similar (small molecule)")
        print_info("GPU benefits increase with larger systems")
    else:
        print_error("GPU is slower than CPU (unexpected)")

def test_larger_molecule():
    """Test 6: Larger Molecule (Benzene)"""
    print_header("Test 6: Larger Molecule Test (Benzene)")

    try:
        from gpu4pyscf import dft as gpu_dft

        # Create benzene molecule
        mol = gto.M(
            atom='''
            C  0.0000  1.3970  0.0000
            C  1.2098  0.6985  0.0000
            C  1.2098 -0.6985  0.0000
            C  0.0000 -1.3970  0.0000
            C -1.2098 -0.6985  0.0000
            C -1.2098  0.6985  0.0000
            H  0.0000  2.4810  0.0000
            H  2.1486  1.2405  0.0000
            H  2.1486 -1.2405  0.0000
            H  0.0000 -2.4810  0.0000
            H -2.1486 -1.2405  0.0000
            H -2.1486  1.2405  0.0000
            ''',
            basis='def2-svp',
            verbose=0
        )

        print_info(f"Molecule: Benzene (C6H6)")
        print_info(f"Basis: {mol.basis}")
        print_info(f"Number of electrons: {mol.nelectron}")
        print_info(f"Number of basis functions: {mol.nao}")

        # GPU calculation
        print_info("Running DFT calculation on GPU...")
        start_time = time.time()

        mf_gpu = gpu_dft.RKS(mol)
        mf_gpu.xc = 'B3LYP'
        mf_gpu.verbose = 0
        energy_gpu = mf_gpu.kernel()

        gpu_time = time.time() - start_time

        print_success(f"Calculation completed in {gpu_time:.3f} seconds")
        print_info(f"Total energy: {energy_gpu:.8f} Hartree")

        # Cleanup GPU calculation objects
        full_cleanup(mol, mf_gpu, verbose=False)

        return True

    except Exception as e:
        print_error(f"Benzene calculation failed: {str(e)}")
        traceback.print_exc()
        return False

def test_gradient_calculation():
    """Test 7: Gradient Calculation"""
    print_header("Test 7: Gradient Calculation")

    try:
        from gpu4pyscf import dft as gpu_dft

        # Simple molecule for gradient test
        mol = gto.M(
            atom='H 0 0 0; H 0 0 0.74',
            basis='def2-svp',
            verbose=0
        )

        print_info(f"Molecule: H2")
        print_info("Computing energy and gradients on GPU...")

        mf_gpu = gpu_dft.RKS(mol)
        mf_gpu.xc = 'B3LYP'
        mf_gpu.verbose = 0
        energy = mf_gpu.kernel()

        grad = mf_gpu.nuc_grad_method()
        forces = grad.kernel()

        print_success("Gradient calculation completed")
        print_info(f"Energy: {energy:.8f} Hartree")
        print_info(f"Forces shape: {forces.shape}")
        print_info(f"Max force: {np.max(np.abs(forces)):.6f} Hartree/Bohr")

        # Cleanup GPU calculation objects
        full_cleanup(mol, mf_gpu, grad, verbose=False)

        return True

    except Exception as e:
        print_error(f"Gradient calculation failed: {str(e)}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  GPU4PySCF Test Suite for RTX 5070 Ti")
    print("=" * 70)

    results = []

    # Test 1: GPU Detection
    results.append(("GPU Detection", test_gpu_detection()))

    # Test 2: GPU4PySCF Import
    results.append(("GPU4PySCF Import", test_gpu4pyscf_import()))

    # Test 3: CPU DFT
    success, energy_cpu, cpu_time, mol = test_basic_dft_cpu()
    results.append(("CPU DFT Calculation", success))

    # Test 4: GPU DFT
    if success and mol is not None:
        success_gpu, gpu_time = test_basic_dft_gpu(mol, energy_cpu)
        results.append(("GPU DFT Calculation", success_gpu))

        # Test 5: Performance
        if success_gpu:
            test_performance_comparison(cpu_time, gpu_time)
    else:
        results.append(("GPU DFT Calculation", False))

    # Test 6: Larger molecule
    results.append(("Larger Molecule (Benzene)", test_larger_molecule()))

    # Test 7: Gradients
    results.append(("Gradient Calculation", test_gradient_calculation()))

    # Summary
    print_header("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30s} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Final cleanup of all GPU memory and cache
    print_header("Final Cleanup")
    cleanup_gpu_memory(verbose=True)

    if passed == total:
        print_success("\nAll tests passed! GPU4PySCF is working correctly.")
        return 0
    else:
        print_error(f"\n{total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
