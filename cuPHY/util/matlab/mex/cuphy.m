% Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

% MATLAB class wrapper to an underlying instance of a wrapper class
% for library function calls.
classdef cuphy < handle
    properties (SetAccess = private, Hidden = false)
        objectHandle;
    end
    methods
        %% -------------------------------------------------------------
        %% Constructor: Create a new instance of the internal wrapper class
        function this = cuphy(varargin)
            this.objectHandle = cuphy_mex('create', varargin{:});
        end
        %% -------------------------------------------------------------
        %% Destructor: Clean up the internal wrapper instance
        function delete(this)
            cuphy_mex('delete', this.objectHandle);
        end
        %% -------------------------------------------------------------
        %% Perform 1D MMSE channel estimation
        function varargout = channelEstMMSE1D(this, varargin)
          [varargout{1:nargout}] = cuphy_mex('channelEstMMSE1D', this.objectHandle, varargin{:});
        end
    end
end
