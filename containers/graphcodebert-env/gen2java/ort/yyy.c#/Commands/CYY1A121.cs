namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: CYY1A121_SERVER_TERMINATION      Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:40:40
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class CYY1A121 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    CYY1A121_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    CYY1A121_OA WOa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // LOCAL VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    CYY1A121_LA WLa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.CYYY9051_OA Cyyy9051Oa;
    GEN.ORT.YYY.CYYY9041_IA Cyyy9041Ia;
    GEN.ORT.YYY.CYYY9041_OA Cyyy9041Oa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // START OF EXIT STATES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public const int ex_StdReturnRb001 = 18988;
    public const int ex_StdReturn002 = 18982;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020239_esc_flag;
    //       +->   CYY1A121_SERVER_TERMINATION       01/09/2024  13:40
    //       !       IMPORTS:
    //       !         Work View imp_dialect iyy1_component (Transient,
    //       !                     Optional, Import only)
    //       !           dialect_cd
    //       !         Work View imp_error iyy1_component (Transient,
    //       !                     Optional, Import only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !       EXPORTS:
    //       !         Work View exp_error_msg iyy1_component (Transient,
    //       !                     Export only)
    //       !           severity_code
    //       !           message_tx
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !       LOCALS:
    //       !         Work View loc_imp_error iyy1_component
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           implementation_id
    //       !           specification_id
    //       !           dialect_cd
    //       !           activity_cd
    //       !           checksum
    //       !         Work View loc_error_msg iyy1_component
    //       !           severity_code
    //       !           message_tx
    //       !         Work View loc_error iyy1_component
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !         Work View loc dont_change_return_codes
    //       !           1_ok
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  Please review explanation for purpose.
    //     2 !   
    //     3 !  NOTE: 
    //     3 !  RELEASE HISTORY
    //     3 !  01_00 23-02-1998 New release
    //     4 !   
    //     5 !  SET loc dont_change_return_codes 1_ok TO 1 
    //     6 !   
    //     7 !  NOTE: 
    //     7 !  ****************************************************************
    //     7 !  Please format the error message.
    //     7 !  
    //     8 !  MOVE imp_error iyy1_component TO loc_imp_error iyy1_component
    //     9 !   
    //    10 !  NOTE: 
    //    10 !  ****************************************************************
    //    10 !  This component is which spec + impl + serv ?
    //    10 !  
    //    11 !  USE cyyy9051_cmpnt_identifiers_get
    //    11 !     WHICH EXPORTS: Work View loc_imp_error iyy1_component FROM
    //    11 !              Work View exp_identifiers iyy1_component
    //    11 !                    Work View loc_error iyy1_component FROM Work
    //    11 !              View exp_error iyy1_component
    //    12 !  +->IF loc_error iyy1_component return_code < loc
    //    12 !  !        dont_change_return_codes 1_ok
    //    13 !  !  MOVE loc_error iyy1_component TO loc_imp_error
    //    13 !  !              iyy1_component
    //    12 !  +--
    //    14 !   
    //    15 !  NOTE: 
    //    15 !  ****************************************************************
    //    15 !  Set the dialect code.
    //    15 !  
    //    16 !  SET loc_imp_error iyy1_component dialect_cd TO imp_dialect
    //    16 !              iyy1_component dialect_cd 
    //    17 !   
    //    18 !  NOTE: 
    //    18 !  ****************************************************************
    //    18 !  Convert the error data to message.
    //    18 !  
    //    19 !  USE cyyy9041_excptn_msg_fmt_as_str
    //    19 !     WHICH IMPORTS: Work View loc_imp_error iyy1_component TO
    //    19 !              Work View imp_error iyy1_component
    //    19 !     WHICH EXPORTS: Work View loc_error_msg iyy1_component FROM
    //    19 !              Work View exp_error_msg iyy1_component
    //    19 !                    Work View loc_error iyy1_component FROM Work
    //    19 !              View exp_error iyy1_component
    //    20 !  +->IF loc_error iyy1_component return_code < loc
    //    20 !  !        dont_change_return_codes 1_ok
    //    21 !  !  MOVE loc_error iyy1_component TO loc_imp_error
    //    21 !  !              iyy1_component
    //    20 !  +--
    //    22 !   
    //    23 !  NOTE: 
    //    23 !  ****************************************************************
    //    23 !  If message is not formatted, use the available data.
    //    23 !  
    //    24 !  +->IF loc_error_msg iyy1_component message_tx <= SPACES
    //    25 !  !  SET loc_error_msg iyy1_component message_tx TO
    //    25 !  !              loc_imp_error iyy1_component context_string 
    //    24 !  +--
    //    26 !  +->IF loc_error_msg iyy1_component severity_code <= SPACES
    //    27 !  !  SET loc_error_msg iyy1_component severity_code TO
    //    27 !  !              loc_imp_error iyy1_component severity_code 
    //    26 !  +--
    //    28 !   
    //    29 !  NOTE: 
    //    29 !  ****************************************************************
    //    29 !  If error code is negative, set Severity = 'Error'
    //    29 !  
    //    30 !  +->IF loc_imp_error iyy1_component return_code < loc
    //    30 !  !        dont_change_return_codes 1_ok
    //    31 !  !  SET loc_imp_error iyy1_component severity_code TO "E" 
    //    32 !  !  SET loc_error_msg iyy1_component severity_code TO
    //    32 !  !              loc_imp_error iyy1_component severity_code 
    //    30 !  +--
    //    33 !   
    //    34 !  MOVE loc_error_msg iyy1_component TO exp_error_msg
    //    34 !              iyy1_component
    //    35 !  MOVE loc_imp_error iyy1_component TO exp_error iyy1_component
    //    36 !   
    //    37 !  +->IF exp_error iyy1_component return_code < loc
    //    37 !  !        dont_change_return_codes 1_ok
    //    38 !  !   
    //    39 !  !  EXIT STATE IS std_return_rb WITH ROLLBACK
    //    40 !  !   
    //    37 !  +> ELSE
    //    41 !  !   
    //    42 !  !  EXIT STATE IS std_return
    //    43 !  !   
    //    37 !  +--
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public CYY1A121(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:40:40";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "CYY1A121_SERVER_TERMINATION";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	CYY1A121_IA import_view, 
    	CYY1A121_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      f_22020239_localAlloc( "CYY1A121_SERVER_TERMINATION" );
      if ( Globdata.GetErrorData().GetLastStatus() == ErrorData.LastStatusIefAllocationError )
      	return;
      
      ++(NestingLevel);
      try {
        f_22020239_init(  );
        f_22020239(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020239(  )
    {
      func_0022020239_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020239" );
      Globdata.GetStateData().SetCurrentABName( "CYY1A121_SERVER_TERMINATION" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    Please review explanation for purpose.                          
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000005" );
      WLa.LocDontChangeReturnCodesQ1Ok = IntAttr.ValueOf((int)TIRD2DEC.Execute1(1, 0, TIRD2DEC.ROUND_NONE, 5));
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    **************************************************************  
      //    **                                                              
      //    Please format the error message.                                
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000008" );
      WLa.LocImpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WIa.ImpErrorIyy1ComponentSeverityCode, 1);
      WLa.LocImpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WIa.ImpErrorIyy1ComponentRollbackIndicator, 1);
      WLa.LocImpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WIa.ImpErrorIyy1ComponentOriginServid);
      WLa.LocImpErrorIyy1ComponentContextString = StringAttr.ValueOf(WIa.ImpErrorIyy1ComponentContextString, 512);
      WLa.LocImpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WIa.ImpErrorIyy1ComponentReturnCode);
      WLa.LocImpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WIa.ImpErrorIyy1ComponentReasonCode);
      WLa.LocImpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(WIa.ImpErrorIyy1ComponentChecksum, 15);
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    **************************************************************  
      //    **                                                              
      //    This component is which spec + impl + serv ?                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000011" );
      
      Cyyy9051Oa = (GEN.ORT.YYY.CYYY9051_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9051).Assembly,
      	"GEN.ORT.YYY.CYYY9051_OA" ));
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY9051).Assembly,
      	"GEN.ORT.YYY.CYYY9051",
      	"Execute",
      	null,
      	Cyyy9051Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020239" );
      Globdata.GetStateData().SetCurrentABName( "CYY1A121_SERVER_TERMINATION" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000011" );
      WLa.LocErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9051Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WLa.LocErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy9051Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WLa.LocErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy9051Oa.ExpErrorIyy1ComponentOriginServid);
      WLa.LocErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy9051Oa.ExpErrorIyy1ComponentContextString, 512);
      WLa.LocErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy9051Oa.ExpErrorIyy1ComponentReturnCode);
      WLa.LocErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy9051Oa.ExpErrorIyy1ComponentReasonCode);
      WLa.LocErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy9051Oa.ExpErrorIyy1ComponentChecksum, 15);
      WLa.LocImpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy9051Oa.ExpIdentifiersIyy1ComponentOriginServid);
      WLa.LocImpErrorIyy1ComponentImplementationId = DoubleAttr.ValueOf(Cyyy9051Oa.ExpIdentifiersIyy1ComponentImplementationId);
      WLa.LocImpErrorIyy1ComponentSpecificationId = DoubleAttr.ValueOf(Cyyy9051Oa.ExpIdentifiersIyy1ComponentSpecificationId);
      Cyyy9051Oa.FreeInstance(  );
      Cyyy9051Oa = null;
      Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
      if ( ((double) WLa.LocErrorIyy1ComponentReturnCode < (double) WLa.LocDontChangeReturnCodesQ1Ok) )
      {
        Globdata.GetStateData().SetLastStatementNumber( "0000000013" );
        WLa.LocImpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentSeverityCode, 1);
        WLa.LocImpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentRollbackIndicator, 1);
        WLa.LocImpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WLa.LocErrorIyy1ComponentOriginServid);
        WLa.LocImpErrorIyy1ComponentContextString = StringAttr.ValueOf(WLa.LocErrorIyy1ComponentContextString, 512);
        WLa.LocImpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WLa.LocErrorIyy1ComponentReturnCode);
        WLa.LocImpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WLa.LocErrorIyy1ComponentReasonCode);
        WLa.LocImpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentChecksum, 15);
      }
      
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    **************************************************************  
      //    **                                                              
      //    Set the dialect code.                                           
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000016" );
      WLa.LocImpErrorIyy1ComponentDialectCd = FixedStringAttr.ValueOf(WIa.ImpDialectIyy1ComponentDialectCd, 2);
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    **************************************************************  
      //    **                                                              
      //    Convert the error data to message.                              
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000019" );
      
      Cyyy9041Ia = (GEN.ORT.YYY.CYYY9041_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9041).Assembly,
      	"GEN.ORT.YYY.CYYY9041_IA" ));
      Cyyy9041Oa = (GEN.ORT.YYY.CYYY9041_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9041).Assembly,
      	"GEN.ORT.YYY.CYYY9041_OA" ));
      Cyyy9041Ia.ImpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WLa.LocImpErrorIyy1ComponentOriginServid);
      Cyyy9041Ia.ImpErrorIyy1ComponentContextString = StringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentContextString, 512);
      Cyyy9041Ia.ImpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WLa.LocImpErrorIyy1ComponentReturnCode);
      Cyyy9041Ia.ImpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WLa.LocImpErrorIyy1ComponentReasonCode);
      Cyyy9041Ia.ImpErrorIyy1ComponentImplementationId = DoubleAttr.ValueOf(WLa.LocImpErrorIyy1ComponentImplementationId);
      Cyyy9041Ia.ImpErrorIyy1ComponentSpecificationId = DoubleAttr.ValueOf(WLa.LocImpErrorIyy1ComponentSpecificationId);
      Cyyy9041Ia.ImpErrorIyy1ComponentDialectCd = FixedStringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentDialectCd, 2);
      Cyyy9041Ia.ImpErrorIyy1ComponentActivityCd = FixedStringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentActivityCd, 15);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY9041).Assembly,
      	"GEN.ORT.YYY.CYYY9041",
      	"Execute",
      	Cyyy9041Ia,
      	Cyyy9041Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020239" );
      Globdata.GetStateData().SetCurrentABName( "CYY1A121_SERVER_TERMINATION" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000019" );
      WLa.LocErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WLa.LocErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WLa.LocErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentOriginServid);
      WLa.LocErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentContextString, 512);
      WLa.LocErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentReturnCode);
      WLa.LocErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentReasonCode);
      WLa.LocErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy9041Oa.ExpErrorIyy1ComponentChecksum, 15);
      WLa.LocErrorMsgIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9041Oa.ExpErrorMsgIyy1ComponentSeverityCode, 1);
      WLa.LocErrorMsgIyy1ComponentMessageTx = StringAttr.ValueOf(Cyyy9041Oa.ExpErrorMsgIyy1ComponentMessageTx, 512);
      Cyyy9041Ia.FreeInstance(  );
      Cyyy9041Ia = null;
      Cyyy9041Oa.FreeInstance(  );
      Cyyy9041Oa = null;
      Globdata.GetStateData().SetLastStatementNumber( "0000000020" );
      if ( ((double) WLa.LocErrorIyy1ComponentReturnCode < (double) WLa.LocDontChangeReturnCodesQ1Ok) )
      {
        Globdata.GetStateData().SetLastStatementNumber( "0000000021" );
        WLa.LocImpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentSeverityCode, 1);
        WLa.LocImpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentRollbackIndicator, 1);
        WLa.LocImpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WLa.LocErrorIyy1ComponentOriginServid);
        WLa.LocImpErrorIyy1ComponentContextString = StringAttr.ValueOf(WLa.LocErrorIyy1ComponentContextString, 512);
        WLa.LocImpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WLa.LocErrorIyy1ComponentReturnCode);
        WLa.LocImpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WLa.LocErrorIyy1ComponentReasonCode);
        WLa.LocImpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentChecksum, 15);
      }
      
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    **************************************************************  
      //    **                                                              
      //    If message is not formatted, use the available data.            
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000024" );
      if ( CompareExit.CompareTo(WLa.LocErrorMsgIyy1ComponentMessageTx, Spaces) <= 0 )
      {
        Globdata.GetStateData().SetLastStatementNumber( "0000000025" );
        WLa.LocErrorMsgIyy1ComponentMessageTx = StringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentContextString, 512);
      }
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000026" );
      if ( CompareExit.CompareTo(WLa.LocErrorMsgIyy1ComponentSeverityCode, Spaces) <= 0 )
      {
        Globdata.GetStateData().SetLastStatementNumber( "0000000027" );
        WLa.LocErrorMsgIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentSeverityCode, 1);
      }
      
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    **************************************************************  
      //    **                                                              
      //    If error code is negative, set Severity = 'Error'               
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000030" );
      if ( ((double) WLa.LocImpErrorIyy1ComponentReturnCode < (double) WLa.LocDontChangeReturnCodesQ1Ok) )
      {
        Globdata.GetStateData().SetLastStatementNumber( "0000000031" );
        WLa.LocImpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf("E", 1);
        Globdata.GetStateData().SetLastStatementNumber( "0000000032" );
        WLa.LocErrorMsgIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentSeverityCode, 1);
      }
      
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000034" );
      WOa.ExpErrorMsgIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocErrorMsgIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorMsgIyy1ComponentMessageTx = StringAttr.ValueOf(WLa.LocErrorMsgIyy1ComponentMessageTx, 512);
      Globdata.GetStateData().SetLastStatementNumber( "0000000035" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WLa.LocImpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WLa.LocImpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WLa.LocImpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(WLa.LocImpErrorIyy1ComponentChecksum, 15);
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000037" );
      if ( ((double) WOa.ExpErrorIyy1ComponentReturnCode < (double) WLa.LocDontChangeReturnCodesQ1Ok) )
      {
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000039" );
        Globdata.GetStateData().SetExitState( ex_StdReturnRb001 );
        Globdata.GetStateData().SetExitInfoMsg( " " );
        Globdata.GetErrorData().SetRollbackRequested( 'R' );
        Globdata.GetStateData().SetExitMsgType( 'N' );
        
      }
      else 
      {
        Globdata.GetStateData().SetLastSubStatementNumber( "1" );
        {
          
          Globdata.GetStateData().SetLastStatementNumber( "0000000042" );
          Globdata.GetStateData().SetExitState( ex_StdReturn002 );
          Globdata.GetStateData().SetExitInfoMsg( " " );
          Globdata.GetErrorData().SetRollbackRequested( ' ' );
          Globdata.GetStateData().SetExitMsgType( 'N' );
          
        }
      }
      
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020239_localAlloc( String abname )
    {
      // Request localview allocation 
      WLa = (GEN.ORT.YYY.CYY1A121_LA)(IefRuntimeParm2.GetInstance( GetType().Assembly,
      	"GEN.ORT.YYY.CYY1A121_LA" ));
      if ( WLa == null )
      {
        Globdata.GetStateData().SetCurrentABId( "0022020239" );
        Globdata.GetStateData().SetCurrentABName( abname );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusIefAllocationError );
      }
    }
    
    public void f_22020239_init(  )
    {
      if ( NestingLevel < 2 )
      {
        WLa.Reset();
      }
      WOa.ExpErrorMsgIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorMsgIyy1ComponentMessageTx = "";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
    }
  }// end class
  
}

