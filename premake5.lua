workspace "Vexora"
    configurations { "Debug", "Release" }
    architecture "x86_64"
    language "C++"
    cppdialect "C++17"
    location "build"

project "Vexora"
    kind "ConsoleApp"
    targetdir "bin/%{cfg.buildcfg}"

    files { 
        "src/**.h", 
        "src/**.cpp", 
    }
    
    includedirs { 
        "include",
        os.getenv("VULKAN_SDK") .. "/Include",
        os.getenv("GLFW_SDK") .. "/include",
    }

    libdirs {
        os.getenv("VULKAN_SDK") .. "/Lib",
        os.getenv("GLFW_SDK") .. "/lib-mingw-w64",
    }

    links {
        "vulkan-1",
        "glfw3",
        "gdi32",
        -- "user32",
    }

    filter "system:windows"
    toolset "gcc"

    filter "configurations:Debug"
        symbols "On"

    filter "configurations:Release"
        optimize "On"
