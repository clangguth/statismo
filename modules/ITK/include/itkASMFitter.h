/*
 * This file is part of the statismo library.
 *
 * Author: Christoph Langguth (christoph.langguth@unibas.ch)
 *
 * Copyright (c) 2011-2015 University of Basel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * Neither the name of the project's author nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef STATISMO_ITKASMFITTER_H
#define STATISMO_ITKASMFITTER_H

#include "ASMFitter.h"

namespace itk {

    class ASMFitterConfiguration : public Object, public statismo::ASMFitterConfiguration {
    public:

        typedef ASMFitterConfiguration Self;
        typedef Object Superclass;
        typedef SmartPointer<Self> Pointer;
        typedef SmartPointer<const Self> ConstPointer;

        itkNewMacro(Self);
        itkTypeMacro(ASMFitterConfiguration, Object);
    };

    template<typename ASM>
    class ASMFitterResult : public Object, public statismo::ASMFitterResult<ASM> {
    public:

        typedef ASMFitterResult Self;
        typedef Object Superclass;
        typedef SmartPointer<Self> Pointer;
        typedef SmartPointer<const Self> ConstPointer;

        itkNewMacro(Self);
        itkTypeMacro(ASMFitterResult, Object);
    };

    //forward declaration for friend class
    template<typename ASM> class ASMFitter;

    /**
     * This class is not meant to be instantiated by end users.
     */
    template<typename ASM>
    class ASMFitterImpl : public Object, public statismo::ASMFitter<ASM> {
    friend class ASMFitter<ASM>;

    public:

        typedef ASMFitterImpl Self;
        typedef Object Superclass;
        typedef SmartPointer <Self> Pointer;
        typedef SmartPointer<const Self> ConstPointer;

        itkTypeMacro( Self, Object);


    protected:

        itkNewMacro( Self );

        virtual typename ASM::FitterResultPointerType NewResult() const {
            return ASM::FitterResultType::New();
        }

        virtual typename ASM::Impl::FitterPointerType NewInstance() const {
            return ASMFitterImpl<ASM>::New();
        }

        virtual typename ASM::Impl::FitterPointerType This() {
            return typename ASM::Impl::FitterPointerType(this);
        }
    };

    template<typename ASM>
    class ASMFitter : public Object {
    public:

        typedef ASMFitter Self;
        typedef Object Superclass;
        typedef SmartPointer <Self> Pointer;
        typedef SmartPointer<const Self> ConstPointer;

        itkNewMacro( Self );
        itkTypeMacro( Self, Object );

        void SetConfiguration(typename ASM::FitterConfigurationPointerType configuration) {
            m_configuration = configuration;
        }

        void SetModel(typename ASM::ActiveShapeModelPointerType model) {
            m_model = model;
        }

        void SetMesh(typename ASM::MeshPointerType inputMesh) {
            m_inputMesh = inputMesh;
        }

        void SetImage(typename ASM::ImagePointerType targetImage) {
            m_targetImage = targetImage;
        }

        void SetSampler(typename ASM::PointSamplerPointerType sampler) {
            m_sampler = sampler;
        }

        void Update() {
            if (!m_impl) {
                m_impl = ASMFitterImpl<ASM>::New();
                m_impl->Init(m_configuration, m_model, m_sampler, m_targetImage, m_inputMesh);
            } else {
                m_impl = ASMFitterImpl<ASM>::New();
                m_impl->Init(m_configuration, m_model, m_sampler, m_targetImage, m_inputMesh);
                //m_impl = m_impl->SetConfiguration(m_configuration)->SetModel(m_model)->SetSampler(m_sampler)->SetImage(m_targetImage)->SetMesh(m_inputMesh);
            }
            m_output = m_impl->Fit();
        }

        typename ASM::FitterResultPointerType GetOutput() {
            return m_output;
        }

    private:
        typename ASM::FitterConfigurationPointerType m_configuration;
        typename ASM::ActiveShapeModelPointerType m_model;
        typename ASM::MeshPointerType m_inputMesh;
        typename ASM::ImagePointerType m_targetImage;
        typename ASM::PointSamplerPointerType m_sampler;
        typename ASM::Impl::FitterPointerType m_impl;
        typename ASM::FitterResultPointerType m_output;
    };

}


#endif //STATISMO_ITKASMFITTER_H
